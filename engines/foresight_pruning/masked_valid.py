# coding=utf-8

import copy

import torch
import wandb
from torch import distributed as dist

import metrics
from engines import BasicValid, ClassificationValid, PruningEngine, SegmentationValid
from tools import named_params, utils

__all__ = ["MaskedValid", "MaskedClassificationValid", "MaskedSegmentationValid"]


class MaskedValid(PruningEngine, BasicValid):
    """The engine is used in the validation phase."""

    def run(self):
        """Valid model"""
        if self.start_epoch != self.config.train.num_epoch:
            return

        self.model.eval()

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

        # Reset metrics
        for v in self.metrics.values():
            v.reset()

        # Synchronizes all processes
        dist.barrier()

        # Run epoch
        valid_dataloader = self.build_dataloader(self.phase)
        valid_losses, valid_metrics = self.run_epoch(valid_dataloader)

        # Save configirations and metrics to files.
        if dist.get_rank() == 0:
            # Weight&bias
            if not self.config.disable_wandb:
                wandb_info = {}

                for k, v in valid_losses.items():
                    wandb_info[f"losses/{self.phase}_{k}"] = v
                for k, v in valid_metrics.items():
                    wandb_info[f"metrics/{self.phase}_{k}"] = v

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
                for k, v in valid_metrics.items():
                    wandb_info[f"density_vs_metric/{self.phase}_{k}"] = wandb.plot.scatter(
                        wandb.Table(
                            data=[[model_density, float(v)]],
                            columns=["model density", f"{self.phase} {k}"],
                        ),
                        x="model density",
                        y=f"{self.phase} {k}",
                        title=f"model density v.s. {self.phase} {k}",
                    )

                wandb.log(wandb_info)

        # Mark the run as finished
        if dist.get_rank() == 0 and not self.config.disable_wandb:
            wandb.finish()


class MaskedClassificationValid(MaskedValid, ClassificationValid):
    """
    `MaskedClassificationValid` is a class that combines the functionalities of `MaskedValid` and
    `ClassificationValid`. It is used for the model validation phase.
    """


class MaskedSegmentationValid(MaskedValid, SegmentationValid):
    """
    `MaskedSegmentationValid` is a class that combines the functionalities of `MaskedValid` and
    `SegmentationValid`. It is used for the model validation phase.
    """
