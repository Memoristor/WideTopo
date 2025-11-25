# coding=utf-8

import copy
import math
import os
import time

import numpy as np
import torch
import wandb
from torch import distributed as dist
from torch.utils.data import DataLoader

from engines import BasicTrain, PruningEngine
from pruners import Pruner
from schedulers import curves
from tools import named_params, utils

__all__ = [
    "ModelPruning",
    "ModelPruningRewind",
]


class ModelPruning(PruningEngine, BasicTrain):
    """The engine is used for foresight model pruning"""

    def __init__(self, *args, **kwags):
        kwags["phase"] = "train"  # use training configs
        super().__init__(*args, **kwags)

    def prune_hook_func(self, dataloader, predict_func, loss_func):
        """Hook function for filling wandb data"""
        return dict()

    def do_pruning(self, dataloader):
        """Do parameter pruning procedure"""
        self.pruning_time = 0

        settings = self.config.foresight_pruning

        # Model pruning iterations
        prune_iters = settings.prune_iters
        for iter in range(1, prune_iters + 1):
            # Pruning start time
            start_time = time.time()

            # Get current model density after pruning
            density = getattr(curves, settings.density_scheduler)(
                x1=0, y1=1, x2=prune_iters, y2=settings.target_density, x=iter
            )

            # Set model mode
            if settings.model_mode == "train":
                self.model.train()
            elif settings.model_mode == "eval":
                self.model.eval()
            else:
                raise AttributeError(f"Unknown model mode {settings.model_mode}, options are `train` and `eval`")

            # Calculate masks
            if settings.mask_type == "global":
                if self.foresight_pruner.__class__.__name__ == "WNT_V10":
                    scores = self.foresight_pruner.calc_score(dataloader, self.predict, self.losses)
                else:
                    self.foresight_pruner.calc_score(dataloader, self.predict, self.losses)
                # self.foresight_pruner.calc_score(dataloader, self.predict, self.losses)
                self.foresight_pruner.global_mask(density=density, tolerance=1.0)
            elif settings.mask_type == "local":
                self.foresight_pruner.calc_score(dataloader, self.predict, self.losses)
                self.foresight_pruner.local_mask(density=density)
            elif settings.mask_type == "custom":
                self.foresight_pruner.custom_mask(
                    dataloader=dataloader,
                    predict_func=self.predict,
                    loss_func=self.losses,
                    density=density,
                )
            else:
                raise AttributeError(
                    f"Unknown mask type {settings.mask_type}, options are `global`, `local`, and `custom`"
                )

            # Apply masks on parameters
            self.foresight_pruner.apply_mask_on_params()

            # Pruning end time
            end_time = time.time()
            self.pruning_time += end_time - start_time

            # Properties
            pruner_props = self.prune_hook_func(dataloader, predict_func=self.predict, loss_func=self.losses)

            # Wandb information
            if dist.get_rank() == 0:
                self.logger.info(
                    f"Pruning model at pruning iteration {iter}/{prune_iters}, density level: {density:4.4f}"
                )

                if not self.config.disable_wandb:
                    tmp_masks = {k: torch.ones_like(p) for k, p in named_params(self.model).items()}
                    for k, m in self.foresight_pruner.masks.items():
                        tmp_masks[k].copy_(m)

                    tmp_masks_flatten = torch.cat([m.flatten() for m in tmp_masks.values()])
                    model_density = tmp_masks_flatten.sum() / tmp_masks_flatten.numel()
                    param_density = {k: m.sum() / m.numel() for k, m in tmp_masks.items()}

                    if (
                        self.foresight_pruner.__class__.__name__ == "WNT_V10"
                        and isinstance(scores, tuple)
                        and len(scores) == 2
                    ):
                        pruner_props["foresight_pruning/ntk_scores"] = scores[0]
                        pruner_props["foresight_pruning/nad_scores"] = scores[1]
                        pruner_props["foresight_pruning/remaining_params"] = tmp_masks_flatten.sum()

                    weights_density = [v for k, v in param_density.items() if ".bias" not in k]
                    collaps_99 = [v for v in weights_density if v <= 0.01]
                    collaps_98 = [v for v in weights_density if v <= 0.02]
                    collaps_95 = [v for v in weights_density if v <= 0.05]
                    pruner_props["layer_collaps/num_weights"] = len(weights_density)
                    pruner_props["layer_collaps/99_percentile"] = 1 - len(collaps_99) / len(weights_density)
                    pruner_props["layer_collaps/98_percentile"] = 1 - len(collaps_98) / len(weights_density)
                    pruner_props["layer_collaps/95_percentile"] = 1 - len(collaps_95) / len(weights_density)

                    wandb.log(
                        {
                            "foresight_pruning_iters": iter,
                            "foresight_pruning/pruning_time": self.pruning_time,
                            "foresight_pruning/pruning_density": density,
                            "foresight_pruning/prunned_density": float(model_density.cpu().numpy()),
                            "foresight_pruning/distribution": wandb.plot.bar(
                                wandb.Table(
                                    data=[
                                        [f"{i + 1:05d}:{k}", float(v.detach().cpu().numpy())]
                                        for i, (k, v) in enumerate(param_density.items())
                                    ],
                                    columns=["param name", "param density"],
                                ),
                                label="param name",
                                value="param density",
                                title="density distribution",
                            ),
                            **pruner_props,
                        }
                    )

    def run(self):
        """
        (Overwrite) Model pruning.

        This method overrides the base class method to implement the pruning logic
        specific to this pruning engine.
        """
        num_epoch = self.config.train.get("num_epoch", None)
        if self.start_epoch >= num_epoch:
            return

        if self.is_resumed:
            return

        # Init weight&bias
        if dist.get_rank() == 0 and not self.config.disable_wandb:
            utils.wandb_init(
                wandb_path=self.save_wandb,
                wandb_config=self.config,
                project_name=self.project_name,
                experiment_name=self.experiment_name,
            )
            wandb.define_metric("foresight_pruning_iters")
            wandb.define_metric("foresight_pruning/*", step_metric="foresight_pruning_iters")

        # Temporarily save initialized parameters
        initialized_params = copy.deepcopy({k: v.data.detach_() for k, v in named_params(self.model).items()})

        # Init model pruner, note that this initialization needs to be done in parallel processes
        # after the model is deployed to the corresponding GPU
        self.foresight_pruner.initialize(self.model)

        # Create dataloader for pruning
        settings = self.config.foresight_pruning
        pruning_dataloader = self.build_dataloader(
            phase="train",
            batch_size=settings.batch_size // dist.get_world_size(),
            num_samples=settings.sample_size,
            # disable_pipeline=True,
        )

        # Pruning at initialization
        if self.start_epoch == 0:
            self.do_pruning(pruning_dataloader)

        # Restore parameters if needed
        if settings.get("restore_params", False):
            for p, init_p in zip(named_params(self.model).values(), initialized_params.values()):
                p.data.copy_(init_p.data)
            self.foresight_pruner.apply_mask_on_params()

        # Check whether the parameters and masks across ranks are consistent
        if dist.get_world_size() > 1:
            masks_vec = torch.cat([torch.clone(m.flatten()).detach() for m in self.foresight_pruner.masks.values()])
            masks_vec_npy = copy.deepcopy(masks_vec.detach().cpu().numpy())
            dist.all_reduce(masks_vec, op=dist.ReduceOp.SUM)
            masks_vec.div_(dist.get_world_size())
            reference_npy = masks_vec.detach().cpu().numpy()
            inconsistent_masks = sum(abs(masks_vec_npy - reference_npy))
            assert inconsistent_masks == 0, f"There are {inconsistent_masks} cross-rank masks are inconsistent"

            params_vec = torch.cat([torch.clone(p.flatten()).detach() for p in named_params(self.model).values()])
            params_vec_npy = copy.deepcopy(params_vec.detach().cpu().numpy())
            dist.all_reduce(params_vec, op=dist.ReduceOp.SUM)
            params_vec.div_(dist.get_world_size())
            reference_npy = params_vec.detach().cpu().numpy()
            if sum(abs(params_vec_npy - reference_npy)) != 0:
                with torch.no_grad():
                    for p in named_params(self.model).values():
                        dist.all_reduce(p, op=dist.ReduceOp.SUM)
                        p.div_(dist.get_world_size())
                if dist.get_rank() == 1:
                    self.logger.info(
                        "The cross-rank parameters are inconsistent, "
                        "and the parameters have been forced to be averaged across ranks."
                    )

        # Save pruned parameters, masks
        if self.start_epoch == 0 and dist.get_rank() == 0:
            self.save(os.path.join(self.save_checkpoints, "last_epoch.pth"), 0)

        # Mark the run as finished
        if dist.get_rank() == 0 and not self.config.disable_wandb:
            wandb.finish()


class ModelPruningRewind(ModelPruning):
    """The engine is used for foresight model pruning with mask restoration"""

    def do_iterative_pruning_once(
        self,
        pruner: Pruner,
        dataloader: DataLoader,
        prune_start_iter: int,
        num_prune_iters: int,
        density_scheduler: str,
        density_scheduler_kwargs: dict,
        model_mode: str,
        mask_type: str,
        mask_tolerance=0.005,
    ):
        """Run pruning procedure iteratively"""

        for iter in range(prune_start_iter, num_prune_iters + prune_start_iter):
            # Pruning start time
            start_time = time.time()

            # Set model mode
            if model_mode == "train":
                self.model.train()
            elif model_mode == "eval":
                self.model.eval()
            else:
                raise AttributeError(f"Unknown model mode {model_mode}, options are `train` and `eval`")

            # Calculate masks
            density = getattr(curves, density_scheduler)(x=iter, **density_scheduler_kwargs)
            if mask_type == "global":
                if pruner.__class__.__name__ == "WNT_V10":
                    scores = pruner.calc_score(dataloader, self.predict, self.losses)
                else:
                    pruner.calc_score(dataloader, self.predict, self.losses)
                pruner.global_mask(density=density, tolerance=mask_tolerance)
            elif mask_type == "local":
                pruner.calc_score(dataloader, self.predict, self.losses)
                pruner.local_mask(density=density, tolerance=mask_tolerance)
            elif mask_type == "custom":
                pruner.custom_mask(
                    dataloader=dataloader,
                    predict_func=self.predict,
                    loss_func=self.losses,
                    density=density,
                    tolerance=mask_tolerance,
                )
            else:
                raise AttributeError(f"Unknown mask type {mask_type}, options are `global`, `local`, and `custom`")

            # Apply masks on parameters
            pruner.apply_mask_on_params()

            # Pruning end time
            end_time = time.time()
            self.pruning_time += end_time - start_time

            # Wandb information
            if dist.get_rank() == 0:
                self.logger.info(f"Pruning model at pruning iteration {iter}, density level: {density:4.4f}")

                if not self.config.disable_wandb:
                    tmp_masks = {k: torch.ones_like(p) for k, p in named_params(self.model).items()}
                    for k, m in pruner.masks.items():
                        tmp_masks[k].copy_(m)

                    tmp_masks_flatten = torch.cat([m.flatten() for m in tmp_masks.values()])
                    model_density = tmp_masks_flatten.sum() / tmp_masks_flatten.numel()
                    param_density = {k: m.sum() / m.numel() for k, m in tmp_masks.items()}

                    pruner_props = {}

                    if pruner.__class__.__name__ == "WNT_V10" and isinstance(scores, tuple) and len(scores) == 2:
                        pruner_props["foresight_pruning/ntk_scores"] = scores[0]
                        pruner_props["foresight_pruning/nad_scores"] = scores[1]
                        pruner_props["foresight_pruning/remaining_params"] = tmp_masks_flatten.sum()

                    weights_density = [v for k, v in param_density.items() if ".bias" not in k]
                    collaps_99 = [v for v in weights_density if v <= 0.01]
                    collaps_98 = [v for v in weights_density if v <= 0.02]
                    collaps_95 = [v for v in weights_density if v <= 0.05]
                    pruner_props["layer_collaps/num_weights"] = len(weights_density)
                    pruner_props["layer_collaps/99_percentile"] = 1 - len(collaps_99) / len(weights_density)
                    pruner_props["layer_collaps/98_percentile"] = 1 - len(collaps_98) / len(weights_density)
                    pruner_props["layer_collaps/95_percentile"] = 1 - len(collaps_95) / len(weights_density)

                    wandb.log(
                        {
                            "foresight_pruning_iters": iter,
                            "foresight_pruning/pruning_time": self.pruning_time,
                            "foresight_pruning/pruning_density": density,
                            "foresight_pruning/prunned_density": float(model_density.cpu().numpy()),
                            "foresight_pruning/distribution": wandb.plot.bar(
                                wandb.Table(
                                    data=[
                                        [f"{i + 1:05d}:{k}", float(v.detach().cpu().numpy())]
                                        for i, (k, v) in enumerate(param_density.items())
                                    ],
                                    columns=["param name", "param density"],
                                ),
                                label="param name",
                                value="param density",
                                title="density distribution",
                            ),
                            **self.prune_hook_func(
                                dataloader,
                                predict_func=self.predict,
                                loss_func=self.losses,
                            ),
                            **pruner_props,
                        }
                    )

    def update_masks_comparison(self, masks_list):
        """Update comparison of masks generated at each pruning iteration"""
        # Masks density and masks similarity
        if dist.get_rank() == 0 and not self.config.disable_wandb:
            wandb_info = {}

            heatmap = np.zeros((len(masks_list), len(masks_list)), dtype=np.float32)
            for i, masks_i in enumerate(masks_list):
                for j, masks_j in enumerate(masks_list):
                    if i < j:
                        masks_i_vec = np.concatenate([m.flatten() for m in masks_i.values()])
                        masks_j_vec = np.concatenate([m.flatten() for m in masks_j.values()])
                        overall_similarity = float(
                            np.array(masks_i_vec == masks_j_vec, dtype=np.float32).sum() / len(masks_i_vec)
                        )
                        masks_similarity = []
                        for idx, (n, m_i) in enumerate(masks_i.items()):
                            m_j = masks_j[n]
                            masks_similarity.append(
                                [
                                    f"{idx + 1:05d}:{n}",
                                    float(np.array(m_i == m_j, dtype=np.float32).sum() / len(m_i.flatten())),
                                ]
                            )
                        wandb_info[f"foresight_pruning/masks_{i}_vs_masks_{j}"] = wandb.plot.bar(
                            wandb.Table(
                                data=masks_similarity,
                                columns=["param name", "mask similarity"],
                            ),
                            label="param name",
                            value="mask similarity",
                            title=f"masks_{i}_vs_masks_{j}",
                        )
                        heatmap[i, j] = overall_similarity
                    elif i == j:
                        heatmap[i, j] = 1.0
                    else:
                        heatmap[i, j] = heatmap[j, i]

                overall_masks = {
                    k: masks_i[k] if k in masks_i.keys() else torch.ones_like(p).detach().cpu().numpy()
                    for k, p in named_params(self.model).items()
                }
                param_density = {k: m.sum() / len(m.flatten()) for k, m in overall_masks.items()}
                wandb_info[f"foresight_pruning/mask {i}"] = wandb.plot.bar(
                    wandb.Table(
                        data=[[f"{i + 1:05d}:{k}", float(v)] for i, (k, v) in enumerate(param_density.items())],
                        columns=["param name", "param density"],
                    ),
                    label="param name",
                    value="param density",
                    title=f"density distribution of mask {i}",
                )

            # xlabels = [f'mask {i}' for i in range(len(masks_list))]
            # wandb_info['foresight_pruning/overall_masks_similarity'] = wandb.plots.HeatMap(
            #     x_labels=xlabels,
            #     y_labels=xlabels,
            #     matrix_values=heatmap,
            #     show_text=True)

            xlabels = [f"mask {i}" for i in range(len(masks_list))]
            wandb_info["foresight_pruning/overall_masks_similarity"] = wandb.Table(
                data=heatmap.tolist(), columns=xlabels
            )

            wandb.log(wandb_info)

    def do_pruning(self, dataloader):
        """Do parameter pruning procedure"""
        self.pruning_time = 0

        foresight_settings = self.config.foresight_pruning
        rewind_settings = self.config.foresight_pruning.rewind

        # Step 0: Obtain initial parameters
        initial_named_params = copy.deepcopy(named_params(self.model))

        # Step 1: Obtain the initial proposal parameter masks
        self.do_iterative_pruning_once(
            pruner=self.foresight_pruner,
            dataloader=dataloader,
            prune_start_iter=1,
            num_prune_iters=foresight_settings.prune_iters,
            density_scheduler=foresight_settings.density_scheduler,
            density_scheduler_kwargs={
                "x1": 0,
                "y1": 1,
                "x2": foresight_settings.prune_iters,
                "y2": foresight_settings.target_density,
            },
            model_mode=foresight_settings.model_mode,
            mask_type=foresight_settings.mask_type,
        )

        masks_list = list()
        masks_list.append({n: m.detach().cpu().numpy() for n, m in self.foresight_pruner.masks.items()})

        # Step 2: Multiple iterations of "Mask Rewind - Iterative Mask Pruning"
        for rwd_iter in range(rewind_settings.rewind_iters):
            # Calculate rewind density
            rewind_compression = -math.log10(foresight_settings.target_density)
            rewind_compression = rewind_compression * (2 ** (rwd_iter + 1) - 1) / (2 ** (rwd_iter + 1))
            rewind_target_density = 10 ** (-rewind_compression)

            # Calculate recoverability
            for rwd_step in range(rewind_settings.rewind_steps):
                # Rewind start time
                start_time = time.time()

                for n, m in self.foresight_pruner.masks.items():
                    s = self.foresight_pruner.scores[n]
                    ones = torch.ones_like(m)
                    if len(m.size()) == 4:  # convolution layers
                        out_deg = 1 - m.sum(dim=[1, 2, 3]) / ones.sum(dim=[1, 2, 3])
                        inp_deg = 1 - m.sum(dim=[0, 2, 3]) / ones.sum(dim=[0, 2, 3])
                        outer = torch.outer(out_deg, inp_deg)
                        outer = torch.unsqueeze(outer, 2)
                        outer = torch.unsqueeze(outer, 3)
                        rand_perb = torch.randn_like(outer) * 1e-5
                        if dist.get_world_size() > 1:
                            dist.all_reduce(rand_perb, op=dist.ReduceOp.SUM)
                            rand_perb.div_(dist.get_world_size())
                        outer = outer + rand_perb
                        s.copy_(torch.clone((1 - m) * outer + m * 2.0).detach())
                    elif len(m.size()) == 2:  # linear layers
                        out_deg = 1 - m.sum(dim=1) / ones.sum(dim=1)
                        inp_deg = 1 - m.sum(dim=0) / ones.sum(dim=0)
                        outer = torch.outer(out_deg, inp_deg)
                        rand_perb = torch.randn_like(outer) * 1e-5
                        if dist.get_world_size() > 1:
                            dist.all_reduce(rand_perb, op=dist.ReduceOp.SUM)
                            rand_perb.div_(dist.get_world_size())
                        outer = outer + rand_perb
                        s.copy_(torch.clone((1 - m) * outer + m * 2.0).detach())
                    else:  # 1-D weight, bias terms
                        s.copy_(torch.clone((1 - m) * ones + m * 2.0).detach())

                # Calculate masks
                rewind_density = curves.linear(
                    x1=0,
                    y1=foresight_settings.target_density,
                    x2=rewind_settings.rewind_steps,
                    y2=rewind_target_density,
                    x=rwd_step + 1,
                )

                mask_type = foresight_settings.mask_type
                if mask_type == "global":
                    self.foresight_pruner.global_mask(density=rewind_density, tolerance=1.0)
                elif mask_type == "local":
                    self.foresight_pruner.local_mask(density=rewind_density, tolerance=1.0)
                elif mask_type == "custom":
                    self.foresight_pruner.custom_mask(
                        dataloader=dataloader,
                        predict_func=self.predict,
                        loss_func=self.losses,
                        density=rewind_density,
                        tolerance=1.0,
                    )
                else:
                    raise AttributeError(f"Unknown mask type {mask_type}, options are `global`, `local` and `custom`")

                # Rewind end time
                end_time = time.time()
                self.pruning_time += end_time - start_time

                # Get model density after rewinding masks
                masks_vec = torch.cat([m.flatten() for m in self.foresight_pruner.masks.values()])
                rewinded_density = masks_vec.sum() / masks_vec.numel()
                if dist.get_rank() == 0 and self.foresight_pruner.verbose:
                    print(f"The model density of rewinding step {rwd_step + 1} is: {rewinded_density:5.5f}")

            # Recover initial parameters
            for p, init_p in zip(self.model.parameters(), initial_named_params.values()):
                p.data.copy_(init_p.data)

            # Apply masks on parameters
            self.foresight_pruner.apply_mask_on_params()

            # Iterative Mask Pruning
            self.do_iterative_pruning_once(
                pruner=self.foresight_pruner,
                dataloader=dataloader,
                prune_start_iter=1 + foresight_settings.prune_iters + rwd_iter * rewind_settings.prune_iters,
                num_prune_iters=rewind_settings.prune_iters,
                density_scheduler=foresight_settings.density_scheduler,
                density_scheduler_kwargs={
                    "x1": foresight_settings.prune_iters + rwd_iter * rewind_settings.prune_iters,
                    "y1": rewinded_density,
                    "x2": foresight_settings.prune_iters + (rwd_iter + 1) * rewind_settings.prune_iters,
                    "y2": foresight_settings.target_density,
                },
                model_mode=foresight_settings.model_mode,
                mask_type=foresight_settings.mask_type,
                mask_tolerance=0.005,
            )

            # Record masks
            masks_list.append({n: m.detach().cpu().numpy() for n, m in self.foresight_pruner.masks.items()})

        # Masks density and masks similarity
        self.update_masks_comparison(masks_list)
