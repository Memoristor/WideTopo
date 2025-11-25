# coding=utf-8

import os

import torch
import yaml
from torch import distributed as dist
from torch.nn.parallel import DataParallel, DistributedDataParallel

import pruners
from engines import BasicEngine
from tools import weight_init

__all__ = ["PruningEngine"]


class PruningEngine(BasicEngine):
    """The pruning engine is built for model pruning training, validation and testing based on BasicEngine

    Params:
        project_name: str. The project name for wandb.
        experiment_name: str. The experiment name in current project.
        config: dict. The runtime environment configuration.
        phase: str. The current phase of the program. Options are `train`, `valid`, and `test`.
        delete_history: bool. A condition to determine whether delete historical experiment data.
    """

    def _prepare_all(self):
        """(Overwrite) Prepare all required attributions"""
        self.prepare_dir()
        self.prepare_logger()
        self.prepare_model()
        self.prepare_optimizer()
        self.prepare_lr_scheduler()
        self.prepare_grad_scaler()
        self.prepare_foresight_pruner()

    def prepare_foresight_pruner(self):
        """Prepare foresight pruner"""
        settings = self.config.foresight_pruning
        if "cls" in settings.pruner:
            with open(settings.prunable_params, "r") as f:
                prunable = yaml.load(f, Loader=yaml.FullLoader)

            assert isinstance(prunable, list)
            self.foresight_pruner = getattr(pruners, settings.pruner.cls)(
                param_names=prunable, verbose=True, **settings.pruner.get("kwargs", {})
            )
        else:
            self.foresight_pruner = None

    def save(self, path, epoch):
        """(Overwrite) Save epoch, model state, optimizer state, and scheduler state.

        Params:
            path: str. The path for saving model.
            epoch: int. The number epoch the model has been trained.
        """
        if dist.get_rank() == 0:
            module_state_dict = {"epoch": epoch}

            # Model state
            if isinstance(self.model, (DataParallel, DistributedDataParallel)):
                state_dict = self.model.module.state_dict()
            else:
                state_dict = self.model.state_dict()
            module_state_dict["model_state_dict"] = state_dict

            # Optimizer state
            if self.optimizer is not None:
                module_state_dict["optimizer_state_dict"] = self.optimizer.state_dict()

            # LR scheduler state
            if self.lr_scheduler is not None:
                module_state_dict["lr_scheduler_state_dict"] = self.lr_scheduler.state_dict()

            # Gradient scaler
            if self.grad_scaler is not None:
                module_state_dict["grad_scaler_state_dict"] = self.grad_scaler.state_dict()

            # Foresight pruner state
            if self.foresight_pruner is not None:
                module_state_dict["foresight_pruner_state_dict"] = self.foresight_pruner.state_dict()

            os.makedirs(os.path.dirname(path), exist_ok=True)
            torch.save(module_state_dict, path)

    def load(self, path):
        """(Overwrite) Load epoch, model state, optimizer state, and scheduler state.

        Params:
            path: str. The file path of saved model parameters.

        Return:
            return the epoch of saved model.
        """
        epoch = 0
        if not os.path.exists(path):
            if dist.get_rank() == 0:
                self.logger.info(f"{self.model_class} can not reload parameters from `{path}`, file not exist")
        else:
            checkpoint = torch.load(path, map_location=torch.device("cpu"))
            epoch = checkpoint["epoch"]

            if dist.get_rank() == 0:
                self.logger.info(f"{self.model_class} reloaded parameters from `{path}`, epoch {epoch}")

            # Load model state dict
            self.model.load_state_dict(checkpoint["model_state_dict"], strict=False)

            # Load optimizer state
            if self.optimizer is not None:
                self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                self.optimizer_to(self.optimizer, torch.device(dist.get_rank()))

            # Load LR scheduler state
            if self.lr_scheduler is not None:
                self.lr_scheduler.load_state_dict(checkpoint["lr_scheduler_state_dict"])

            # Load Gradient scaler state
            if self.grad_scaler is not None:
                self.grad_scaler.load_state_dict(checkpoint["grad_scaler_state_dict"])

            # Load foresight pruner state
            if self.foresight_pruner is not None:
                self.foresight_pruner.load_state_dict(checkpoint["foresight_pruner_state_dict"])

                # Load pruner scores tensors to GPU
                for k, v in self.foresight_pruner.scores.items():
                    if isinstance(v, torch.Tensor):
                        self.foresight_pruner.scores[k] = v.to(torch.device(dist.get_rank()))

                # Load pruner masks tensors to GPU
                for k, v in self.foresight_pruner.masks.items():
                    if isinstance(v, torch.Tensor):
                        self.foresight_pruner.masks[k] = v.to(torch.device(dist.get_rank()))

            # Set resumed states
            self.is_resumed = True

        # Weight initilization if needed
        if epoch == 0 and not self.is_resumed:
            settings = self.config.foresight_pruning
            if "weight_init" in settings.keys():
                self.model.apply(getattr(weight_init, settings.weight_init))
            else:
                if dist.get_rank() == 0:
                    self.logger.info("Weight initilization is not employed.")

        self.distributed_dataparallel()
        return epoch
