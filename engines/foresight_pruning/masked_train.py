# coding=utf-8

import copy
import json
import os

import torch
import wandb
from torch import distributed as dist

import metrics
from engines import BasicTrain, ClassificationTrain, PruningEngine, SegmentationTrain
from tools import named_params, utils

__all__ = ["MaskedTrain", "MaskedClassificationTrain", "MaskedSegmentationTrain"]


class MaskedTrain(PruningEngine, BasicTrain):
    """The engine is used in the training phase after model pruning."""

    def gradient_backward(self, loss_all):
        """(Overwrite) Gradient backward"""
        if self.foresight_pruner is not None:
            # Gradient descent with masks
            self.optimizer.zero_grad()
            # loss_all.backward()
            self.grad_scaler.scale(loss_all).backward()
            # self.foresight_pruner.apply_mask_on_grads()

            # Gradient clipping
            if "gradient_norm_clipping" in self.config.optimizer.keys():
                # Unscales the gradients of optimizer's assigned params in-place
                # https://pytorch.org/docs/stable/notes/amp_examples.html#gradient-clipping
                self.grad_scaler.unscale_(self.optimizer)

                # Since the gradients of optimizer's assigned params are unscaled, clips as usual:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    max_norm=self.config.optimizer.gradient_norm_clipping,
                )

            # self.optimizer.step()
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
            self.foresight_pruner.apply_mask_on_params()
        else:
            super().gradient_backward(loss_all=loss_all)

    def run(self):
        """(Overwrite) Train model"""
        if self.start_epoch >= self.config.train.num_epoch:
            return

        # Init weight&bias
        if dist.get_rank() == 0 and not self.config.disable_wandb:
            utils.wandb_init(
                wandb_path=self.save_wandb,
                wandb_config=self.config,
                project_name=self.project_name,
                experiment_name=self.experiment_name,
            )
            wandb.define_metric("epoch")
            wandb.define_metric("iteration")
            wandb.define_metric("losses/*", step_metric="epoch")
            wandb.define_metric("metrics/*", step_metric="epoch")
            wandb.define_metric("learning_rate/*", step_metric="epoch")
            wandb.define_metric("model_density/*", step_metric="epoch")

        if self.foresight_pruner is not None:
            # Init model pruner, note that this initialization needs to be done in parallel processes
            # after the model is deployed to the corresponding GPU
            if self.foresight_pruner.is_initialized and self.foresight_pruner._model is None:
                self.foresight_pruner._model = self.model

        # Load the best valid metric data
        best_valid_path = os.path.join(self.save_checkpoints, "best_valid.json")
        try:
            with open(best_valid_path, "r") as f:
                best_valid = json.loads(f.read())

            for k in self.metrics.keys():
                if k not in best_valid.keys():
                    best_valid[k] = {"value": 0.0, "epoch": 0}
        except:
            best_valid = dict([(k, {"value": 0.0, "epoch": 0}) for k in self.metrics.keys()])

        # Create dataloader
        train_dataloader = self.build_dataloader(phase="train")
        valid_dataloader = self.build_dataloader(phase="valid")

        # Start training
        for epoch in range(self.start_epoch + 1, self.config.train.num_epoch + 1):
            # Log information
            last_lr = self.lr_scheduler.get_lr()[0]
            if dist.get_rank() == 0:
                self.logger.info(f"[Train] exp name: {self.experiment_name}")
                self.logger.info(f"[Train] current learning rate: {last_lr:6.6f}")

            # Synchronizes all processes
            dist.barrier()

            # Run train epoch and valid epoch
            train_loss, train_metric = self.run_epoch(
                train_dataloader,
                epoch=epoch,
                phase="train",
                mixup=self.mixup,
                summary_freq=self.config.train.summary_freq,
            )
            valid_loss, valid_metric = self.run_epoch(
                valid_dataloader,
                epoch=epoch,
                phase="valid",
                summary_freq=self.config.valid.summary_freq,
            )

            # Save and record result
            if dist.get_rank() == 0:
                if not self.config.disable_wandb:
                    # Add epoch, learning rate, losses and metrics to wandb
                    wandb_info = {
                        "epoch": epoch,
                        "learning_rate/last_lr": last_lr,
                    }

                    for k, v in train_loss.items():
                        wandb_info[f"losses/train_{k}"] = v
                    for k, v in valid_loss.items():
                        wandb_info[f"losses/valid_{k}"] = v
                    for k, v in train_metric.items():
                        wandb_info[f"metrics/train_{k}"] = v
                    for k, v in valid_metric.items():
                        wandb_info[f"metrics/valid_{k}"] = v

                    if self.foresight_pruner is not None:
                        tmp_masks = copy.deepcopy(self.foresight_pruner.masks)
                        for k, p in named_params(self.model).items():
                            if k not in tmp_masks.keys():
                                tmp_masks[k] = torch.ones_like(p)
                        tmp_masks_flatten = torch.cat([m.flatten() for m in tmp_masks.values()])
                        model_density = tmp_masks_flatten.sum() / tmp_masks_flatten.numel()
                    else:
                        model_density = metrics.model_density(self.model, named_parameters=None)

                    model_density = float(model_density.cpu().detach().numpy())
                    wandb_info["model_density/density"] = model_density

                    # Add wandb tables (density vs metrics) at end of training
                    if epoch == self.config.train.num_epoch:
                        for k, v in valid_metric.items():
                            wandb_info[f"density_vs_metric/valid_{k}"] = wandb.plot.scatter(
                                wandb.Table(
                                    data=[[model_density, float(v)]],
                                    columns=["model density", f"valid {k}"],
                                ),
                                x="model density",
                                y=f"valid {k}",
                                title=f"model density v.s. valid {k}",
                            )

                            wandb_info[f"density_vs_metric/best_{k}"] = wandb.plot.scatter(
                                wandb.Table(
                                    data=[
                                        [
                                            model_density,
                                            max(float(v), float(best_valid[k]["value"])),
                                        ]
                                    ],
                                    columns=["model density", f"best {k}"],
                                ),
                                x="model density",
                                y=f"best {k}",
                                title=f"model density v.s. best {k}",
                            )

                    wandb.log(wandb_info)

                # Save lastest model every 10 epochs
                if epoch == 1 or epoch % 10 == 0 or epoch == self.config.train.num_epoch:
                    self.save(os.path.join(self.save_checkpoints, "last_epoch.pth"), epoch)

                # Save best model and corresponding metric results
                for k, v in valid_metric.items():
                    if best_valid[k]["value"] < v:
                        best_valid[k]["value"] = float(v)
                        best_valid[k]["epoch"] = epoch

                        # shutil.copyfile(os.path.join(self.save_checkpoints, 'last_epoch.pth'),
                        #                 os.path.join(self.save_checkpoints, f'best_{k}.pth'))

                        with open(os.path.join(self.save_checkpoints, f"best_{k}.txt"), "w") as f:
                            f.write(
                                f"[Valid] Model: {self.model_class}, Optimizer: {self.config.optimizer.cls}, Epoch: {epoch}\n"
                            )
                            f.write("\n")
                            f.write("[Valid] The train losses\n")
                            for n, s in train_loss.items():
                                f.write(f"{n}: {s:4.4f}\n")
                            f.write("\n")
                            f.write("[Valid] The train metric\n")
                            for n, s in train_metric.items():
                                f.write(f"{n}: {s:4.4f}\n")
                            f.write("\n")
                            f.write("[Valid] The valid losses\n")
                            for n, s in valid_loss.items():
                                f.write(f"{n}: {s:4.4f}\n")
                            f.write("\n")
                            f.write("[Valid] The valid metric\n")
                            for n, s in valid_metric.items():
                                f.write(f"{n}: {s:4.4f}\n")
                            f.write("\n")

                    self.logger.info(
                        f"[Valid] {k}: {valid_metric[k]:5.5f}, best {k}: {best_valid[k]['value']:5.5f}/{best_valid[k]['epoch']}"
                    )

                # Save the best valid metrics
                with open(best_valid_path, "w") as f:
                    f.write(json.dumps(best_valid))

        # Mark the run as finished
        if dist.get_rank() == 0 and not self.config.disable_wandb:
            wandb.finish()


class MaskedClassificationTrain(MaskedTrain, ClassificationTrain):
    """
    `MaskedClassificationTrain` is a class that combines the functionalities of `MaskedTrain` and
    `ClassificationTrain`. It is used in the training phase after model pruning.
    """


class MaskedSegmentationTrain(MaskedTrain, SegmentationTrain):
    """
    `MaskedSegmentationTrain` is a class that combines the functionalities of `MaskedTrain` and
    `SegmentationTrain`. It is used in the training phase after model pruning.
    """
