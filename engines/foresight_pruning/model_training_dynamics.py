# coding=utf-8

import copy
import itertools
import math

import torch
import wandb
from torch import distributed as dist
from tqdm import tqdm

import tools
from engines import BasicTrain, PruningEngine
from tools import utils

__all__ = ["ModelTrainingDynamics"]


class ModelTrainingDynamics(PruningEngine, BasicTrain):
    """The engine is used in the training phase after model pruning.

    Params:
        config: dict. The runtime environment configuration.
    """

    def __init__(self, *args, **kwags):
        super().__init__(*args, **kwags)

        # Add new config for training dynamic check
        if self.config.dataset.cls == "TinyImageNetDataset":
            self.config.training_dynamics = {
                "samples": 5,
                # 'grads_block_size': 750,
                # 'ntk_block_size': 750,
                "hard_momemtum": False,
                "mode": "train_only",  # train_only, eval_only, train_then_eval
            }
        elif self.config.dataset.cls == "ImageNetDataset":
            self.config.training_dynamics = {
                "samples": 1,
                # 'grads_block_size': 750,
                # 'ntk_block_size': 750,
                "hard_momemtum": False,
                "mode": "train_only",  # train_only, eval_only, train_then_eval
            }
        else:
            raise AttributeError("Unsupported dataset !")

    def run(self):
        """(Overwrite) Run this"""
        if self.config.foresight_pruning.pruner.keyword != "NO_PRUNER" and not self.is_resumed:
            raise AttributeError("No checkpoint found !")

        # Init weight&bias
        if dist.get_rank() == 0 and not self.config.disable_wandb:
            utils.wandb_init(
                wandb_path=self.save_wandb,
                wandb_config=self.config,
                project_name=self.project_name,
                experiment_name=self.experiment_name,
            )
            wandb.define_metric("training_dynamics_v2")

        if self.foresight_pruner is not None:
            # Init model pruner, note that this initialization needs to be done in parallel processes
            # after the model is deployed to the corresponding GPU
            if self.foresight_pruner.is_initialized and self.foresight_pruner._model is None:
                self.foresight_pruner._model = self.model

            # Apply masks on model
            self.foresight_pruner.apply_mask_on_params()

        # Get masks vector
        masks = {k: torch.ones_like(p) for k, p in tools.named_params(self.model).items()}
        if self.foresight_pruner is not None:
            for k, m in self.foresight_pruner.masks.items():
                masks[k].copy_(m)
        masks_vec = torch.cat([m.flatten() for m in masks.values()])

        # Run model for once
        num_samples = self.config.training_dynamics.samples
        dataloader = self.build_dataloader(
            phase="train",
            batch_size=num_samples // dist.get_world_size(),
            num_samples=num_samples,
            # disable_pipeline=True,
        )

        truth = next(iter(dataloader))
        for k, v in truth.items():
            truth[k] = v.to(torch.device(dist.get_rank()))

        # Hard BN momentum
        hard_momentum = self.config.training_dynamics.hard_momemtum
        if hard_momentum:
            tools.hard_bn_momentum(self.model)
            tools.reset_bn_status(self.model)

        model_mode = self.config.training_dynamics.mode
        if model_mode == "train_only":
            self.model.train()
            pred = self.predict(model=self.model, truth=truth)
        elif model_mode == "eval_only":
            self.model.eval()
            pred = self.predict(model=self.model, truth=truth)
        elif model_mode == "train_then_eval":
            self.model.train()
            pred = self.predict(model=self.model, truth=truth)
            self.model.eval()
            pred = self.predict(model=self.model, truth=truth)
        else:
            raise AttributeError("Unknown model model!")

        flatten_pred = torch.cat([v.flatten() for v in pred.values()])

        if "grads_block_size" in self.config.training_dynamics.keys():
            dim_ntk = len(flatten_pred)
            grads_block_size = self.config.training_dynamics.grads_block_size
            # assert dim_ntk % grads_block_size == 0

            ntk = torch.zeros(dim_ntk, dim_ntk)
            for i in range(0, int(math.ceil(dim_ntk / grads_block_size))):
                elems_i = flatten_pred[i * grads_block_size : (i + 1) * grads_block_size]

                grad_vecs_i = list()
                bar_i = tqdm(elems_i)
                for elem in bar_i:
                    bar_i.set_description(
                        f"- Grads row: {i + 1} / {int(math.ceil(dim_ntk / grads_block_size))}, calculating gradients with respect to parameters"
                    )
                    self.model.zero_grad()
                    elem.backward(retain_graph=True)
                    grad_vec = torch.cat([p.grad.flatten() for p in self.model.parameters() if p.grad is not None])
                    grad_vec = grad_vec * masks_vec
                    grad_vec = copy.deepcopy(grad_vec.detach().cpu())
                    grad_vecs_i.append(grad_vec)
                grad_vecs_i = torch.stack(grad_vecs_i)

                for j in range(i, int(math.ceil(dim_ntk / grads_block_size))):
                    elems_j = flatten_pred[j * grads_block_size : (j + 1) * grads_block_size]

                    if j != i:
                        grad_vecs_j = list()
                        bar_j = tqdm(elems_j)
                        for elem in bar_j:
                            bar_j.set_description(
                                f"-- Grads col: {j + 1} / {int(math.ceil(dim_ntk / grads_block_size))}, calculating gradients with respect to parameters"
                            )
                            self.model.zero_grad()
                            elem.backward(retain_graph=True)
                            grad_vec = torch.cat(
                                [p.grad.flatten() for p in self.model.parameters() if p.grad is not None]
                            )
                            grad_vec = grad_vec * masks_vec
                            grad_vec = copy.deepcopy(grad_vec.detach().cpu())
                            grad_vecs_j.append(grad_vec)
                        grad_vecs_j = torch.stack(grad_vecs_j)

                    else:
                        grad_vecs_j = grad_vecs_i

                    # calculating ntk block
                    if "ntk_block_size" in self.config.training_dynamics.keys():
                        ntk_block_size = self.config.training_dynamics.ntk_block_size
                        assert len(elems_i) % ntk_block_size == 0 and len(elems_j) % ntk_block_size == 0

                        ntk_block = torch.zeros(len(elems_i), len(elems_j))
                        sub_block_ids = itertools.product(
                            range(0, len(elems_i) // ntk_block_size),
                            range(0, len(elems_j) // ntk_block_size),
                        )
                        bar_pq = tqdm(list(sub_block_ids))
                        for p, q in bar_pq:
                            bar_pq.set_description("--- Computing NTK matrix in a block-wise manner")
                            ntk_block[
                                p * ntk_block_size : (p + 1) * ntk_block_size,
                                q * ntk_block_size : (q + 1) * ntk_block_size,
                            ] = torch.einsum(
                                "ik,jk->ij",
                                grad_vecs_i[p * ntk_block_size : (p + 1) * ntk_block_size, :],
                                grad_vecs_j[q * ntk_block_size : (q + 1) * ntk_block_size, :],
                            )

                    else:
                        ntk_block = torch.einsum("ik,jk->ij", grad_vecs_i, grad_vecs_j)

                    ntk[
                        i * grads_block_size : min((i + 1) * grads_block_size, dim_ntk),
                        j * grads_block_size : min((j + 1) * grads_block_size, dim_ntk),
                    ] = ntk_block

            # Generate a symmetric matrix
            for i in range(0, dim_ntk):
                for j in range(i, dim_ntk):
                    ntk[j, i] = ntk[i, j]

        else:
            grad_vecs = list()
            bar = tqdm(flatten_pred)
            for elem in bar:
                self.model.zero_grad()
                elem.backward(retain_graph=True)
                grad_vec = torch.cat([p.grad.flatten() for p in self.model.parameters() if p.grad is not None])
                grad_vec = grad_vec * masks_vec
                grad_vec = copy.deepcopy(grad_vec.detach().cpu())
                grad_vecs.append(grad_vec)
                bar.set_description("Calculating gradients with respect to parameters")
            grad_vecs = torch.stack(grad_vecs)
            ntk = torch.einsum("ik,jk->ij", grad_vecs, grad_vecs)

        print(ntk)
        eigvals = torch.linalg.eigvalsh(ntk)
        # ntk = ntk.detach().cpu().numpy()
        # eigvals = scipy.linalg.eigvalsh(ntk)

        if dist.get_rank() == 0 and not self.config.disable_wandb:
            wandb.log(
                {
                    f"training_dynamics_v2/{len(eigvals)}_eigvals (hard momemtm: {hard_momentum}, model mode: {model_mode})": wandb.plot.line(
                        wandb.Table(
                            data=[[i, max(float(v), 1e-20)] for i, v in enumerate(eigvals)],
                            columns=["index", "eigval"],
                        ),
                        x="index",
                        y="eigval",
                        title=f"eigval distribution with {len(eigvals)} eigvals (hard momemtm: {hard_momentum}, model mode: {model_mode})",
                    )
                }
            )

        # Reset BN momentum
        tools.reset_bn_momentum(self.model)

        # Mark the run as finished
        if dist.get_rank() == 0 and not self.config.disable_wandb:
            wandb.finish()
