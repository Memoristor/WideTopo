# coding=utf-8

import copy
import os
import shutil
import time
from typing import Optional

import torch
from omegaconf import DictConfig, OmegaConf
from torch import distributed as dist
from torch import optim
from torch.cuda.amp import GradScaler
from torch.nn import SyncBatchNorm
from torch.nn.parallel import DataParallel, DistributedDataParallel
from torch.utils.data import DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler

import datasets
import models
import schedulers
from datasets import dali_pipeline, transforms
from datasets.base_datasets.classification import ClassificationDataset
from datasets.base_datasets.segmentation import SegmentationDataset
from tools import get_logger, named_params

__all__ = ["BasicEngine"]


class BasicEngine:
    """The basic engine is all you need :).

    Params:
        project_name (str): The name of the project for Weights & Biases (wandb).
        experiment_name (str): The name of the experiment within the current project.
        config (dict, DictConfig): The runtime environment configuration as a dictionary.
        phase (str): The current phase of the program. Options are `train`, `valid`, or `test`.
        delete_history (bool): Determines whether to delete historical experiment data.
    """

    def __init__(
        self,
        project_name: str,
        experiment_name: str,
        config: DictConfig,
        phase: str,
        delete_history: bool = False,
    ):
        self.project_name = project_name
        self.experiment_name = experiment_name
        self.config = config
        self.phase = phase
        self.delete_history = delete_history

        self.runtime = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

        # Prepare required attributions
        self._prepare_all()

        # Load checkpoint
        model_pth = os.path.join(self.save_checkpoints, self.config[self.phase].reload_pth)
        self.is_resumed = False
        self.start_epoch = self.load(model_pth)

    def _prepare_all(self):
        """Prepare all required attributions"""
        self.prepare_dir()
        self.prepare_logger()
        self.prepare_model()
        self.prepare_optimizer()
        self.prepare_lr_scheduler()
        self.prepare_grad_scaler()

    def prepare_dir(self):
        """Prepare and make needed directories"""
        self.model_class = self.config.model.cls
        self.dataset_class = self.config.dataset.cls

        output_root = self.config.globals.output_root
        output_dir = os.path.join(
            output_root,
            self.project_name,
            self.dataset_class,
            self.model_class,
            self.experiment_name,
        )

        self.save_checkpoints = os.path.join(output_dir, "checkpoints")
        self.save_wandb = os.path.join(output_dir, "wandb")
        self.save_logger = os.path.join(output_dir, "logs")
        self.save_result = os.path.join(output_dir, "results")

        if self.delete_history:
            if dist.get_rank() == 0:
                print(
                    "Warning! Historical experimental data will be deleted in 15 seconds! "
                    "Press Ctrl+C if you want this stop!"
                )
                for remaining in range(15, 0, -1):
                    print(f"\rCounting down: {remaining} seconds", end="")
                    time.sleep(1)

            shutil.rmtree(self.save_checkpoints, ignore_errors=True)
            shutil.rmtree(self.save_wandb, ignore_errors=True)
            shutil.rmtree(self.save_logger, ignore_errors=True)
            shutil.rmtree(self.save_result, ignore_errors=True)

        os.makedirs(self.save_checkpoints, exist_ok=True)
        os.makedirs(self.save_wandb, exist_ok=True)
        os.makedirs(self.save_logger, exist_ok=True)
        os.makedirs(self.save_result, exist_ok=True)

    def prepare_logger(self):
        """Prepare and make required logger"""
        self.logger = get_logger.Logger(
            filename=os.path.join(
                self.save_logger, f"rank_{dist.get_rank()}_runtime_{self.runtime}.log"
            ),
            level=self.config.globals.log_level,
        ).logger

    def prepare_optimizer(self):
        """Prepare and make required optimizer"""
        if "optimizer" in self.config.keys():
            self.optimizer = getattr(optim, self.config.optimizer.cls)(
                params=self.model.parameters(),
                **self.config.optimizer.get("kwargs", {}),
            )
        else:
            self.optimizer = None

    def prepare_model(self):
        """Prepare model"""
        self.model = getattr(models, self.model_class)(**self.config.model.get("kwargs", {}))

    def prepare_lr_scheduler(self):
        """Prepare learning rate scheduler"""
        if "learning_rate" in self.config.keys():
            if self.config.learning_rate.warmup_steps > 0:
                self.lr_scheduler = schedulers.GradualWarmupScheduler(
                    optimizer=self.optimizer,
                    multiplier=1.0,
                    total_epoch=self.config.learning_rate.warmup_steps,
                    after_scheduler=getattr(schedulers, self.config.learning_rate.cls)(
                        optimizer=self.optimizer,
                        **self.config.learning_rate.get("kwargs", {}),
                    ),
                )
            else:
                self.lr_scheduler = getattr(schedulers, self.config.learning_rate.cls)(
                    optimizer=self.optimizer,
                    **self.config.learning_rate.get("kwargs", {}),
                )
        else:
            self.lr_scheduler = None

    def prepare_grad_scaler(self):
        """Prepare gradient scaler"""
        self.grad_scaler = GradScaler()
        self.grad_scaler._enabled = self.config.train.enable_amp

    def distributed_dataparallel(self):
        """Prepare distributed data parallel"""
        # Load model to GPU
        self.model = self.model.to(torch.device(dist.get_rank()))

        # Check whether the parameters across ranks are consistent
        if dist.get_world_size() > 1:
            params_vec = torch.cat(
                [torch.clone(p.flatten()).detach() for p in named_params(self.model).values()]
            )
            params_vec_npy = copy.deepcopy(params_vec.detach().cpu().numpy())
            dist.all_reduce(params_vec, op=dist.ReduceOp.SUM)
            params_vec.div_(dist.get_world_size())
            reference_npy = params_vec.detach().cpu().numpy()
            if sum(abs(params_vec_npy - reference_npy)) != 0:
                with torch.no_grad():
                    for p in named_params(self.model).values():
                        dist.all_reduce(p, op=dist.ReduceOp.SUM)
                        p.div_(dist.get_world_size())
                if dist.get_rank() == 0:
                    self.logger.info(
                        "The cross-rank parameters are inconsistent, "
                        "and the parameters have been forced to be averaged across ranks."
                    )

        # Convert batchnorm to syncBN if needed
        if self.config.convert_syncbn:
            self.model = SyncBatchNorm.convert_sync_batchnorm(self.model)

        # Distributed Data Parallel
        self.model = DistributedDataParallel(
            self.model,
            device_ids=[dist.get_rank()],
            output_device=dist.get_rank(),
            # find_unused_parameters=True,
        )

    def optimizer_to(self, optim, device):
        """Deploy optimizer to device"""
        for param in optim.state.values():
            # Not sure there are any global tensors in the state dict
            if isinstance(param, torch.Tensor):
                param.data = param.data.to(device)
                if param._grad is not None:
                    param._grad.data = param._grad.data.to(device)
            elif isinstance(param, dict):
                for subparam in param.values():
                    if isinstance(subparam, torch.Tensor):
                        subparam.data = subparam.data.to(device)
                        if subparam._grad is not None:
                            subparam._grad.data = subparam._grad.data.to(device)

    def save(self, path, epoch):
        """Save epoch, model state, optimizer state, and scheduler state.

        Params:
            path (str): The path for saving model.
            epoch (int): The number epoch the model has been trained.
        """
        if dist.get_rank() == 0:
            if isinstance(self.model, DataParallel) or isinstance(
                self.model, DistributedDataParallel
            ):
                state_dict = self.model.module.state_dict()
            else:
                state_dict = self.model.state_dict()

            os.makedirs(os.path.dirname(path), exist_ok=True)
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": state_dict,
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "lr_scheduler_state_dict": self.lr_scheduler.state_dict(),
                    "grad_scaler_state_dict": self.grad_scaler.state_dict(),
                },
                path,
            )

    def load(self, path):
        """Load epoch, model state, optimizer state, and scheduler state.

        Params:
            path (str): The file path of saved model parameters.

        Returns:
            return the epoch of saved model.
        """
        epoch = 0
        if not os.path.exists(path):
            if dist.get_rank() == 0:
                self.logger.info(
                    f"{self.model_class} can not reload parameters from `{path}`, file not exist"
                )
        else:
            checkpoint = torch.load(path, map_location=torch.device("cpu"))
            epoch = checkpoint["epoch"]

            if dist.get_rank() == 0:
                self.logger.info(
                    f"{self.model_class} reloaded parameters from `{path}`, epoch {epoch}"
                )

            # Load model state dict
            self.model.load_state_dict(checkpoint["model_state_dict"], strict=False)
            self.lr_scheduler.load_state_dict(checkpoint["lr_scheduler_state_dict"])
            if "grad_scaler_state_dict" in checkpoint.keys():
                self.grad_scaler.load_state_dict(checkpoint["grad_scaler_state_dict"])
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            self.optimizer_to(self.optimizer, torch.device(dist.get_rank()))

            # Set resumed states
            self.is_resumed = True

        self.distributed_dataparallel()
        return epoch

    def build_dataloader(
        self,
        phase: str,
        batch_size: Optional[int] = None,
        num_samples: Optional[int] = None,
        disable_pipeline: bool = False,
    ):
        """Build distributed Pytorch Dataloader or DALI Pipeloader

        Params:
            phase (str): The phase for dataset configurations.
            batch_size (int | None): The batch size of the dataloader.
            num_samples (int | None): The number of samples randomly sampled.
            disable_pipeline (bool): Disable DALI pipeline when the pipeline is available.

        Returns:
            An iterable data loader.
        """
        # Build transforms
        config = self.config[phase]
        transformer = None
        if "transforms" in config.dataset.keys():
            transformer = getattr(transforms, config.dataset.transforms.func)(
                **config.dataset.transforms.get("kwargs", {})
            )

        # Build dataset
        dataset = getattr(datasets, self.config.dataset.cls)(
            root_path=self.config.dataset.root_path,
            transforms=transformer,
            **config.dataset.get("kwargs", {}),
        )
        assert len(dataset) > 0, "The dataset is empty."

        # Build iterable dataloader
        num_workers = self.config.dataset.num_workers
        if batch_size is None:
            batch_size = config.batch_size // dist.get_world_size()

        if "pipeline" in config.dataset.keys() and not disable_pipeline:
            if isinstance(dataset, ClassificationDataset):
                files = dataset.image_path
                if num_samples is not None:
                    indices = [int(i) for i in torch.randperm(len(dataset))[:num_samples]]
                    files = [files[i] for i in indices]
                labels = [dataset.load_label(f) for f in files]

                pipeline_kwargs = config.dataset.pipeline.get("kwargs", {})
                if len(pipeline_kwargs) > 0:
                    pipeline_kwargs = OmegaConf.to_container(pipeline_kwargs)

                pipeline = getattr(dali_pipeline, config.dataset.pipeline.func)(
                    batch_size=batch_size,
                    num_threads=num_workers,
                    device_id=dist.get_rank(),
                    files=files,
                    labels=labels,
                    num_shards=dist.get_world_size(),
                    shard_id=dist.get_rank(),
                    random_seed=self.config.random_seed + dist.get_rank(),
                    **pipeline_kwargs,
                )
                pipeline.build()

                return dali_pipeline.ImagenetPipeloader(
                    dataset=dataset, pipeline=pipeline, reader_name=f"Reader{dist.get_rank()}"
                )

            elif isinstance(dataset, SegmentationDataset):
                indices = None
                if num_samples is not None:
                    indices = [int(i) for i in torch.randperm(len(dataset))[:num_samples]]

                pipeline_kwargs = config.dataset.pipeline.get("kwargs", {})
                if len(pipeline_kwargs) > 0:
                    pipeline_kwargs = OmegaConf.to_container(pipeline_kwargs)

                pipeline = getattr(dali_pipeline, config.dataset.pipeline.func)(
                    dataset=dataset,
                    batch_size=batch_size,
                    batch_size_extsrc=batch_size,  # must be same as `batch_size`
                    sampling_indices=indices,
                    num_threads=num_workers,
                    device_id=dist.get_rank(),
                    num_shards=dist.get_world_size(),
                    shard_id=dist.get_rank(),
                    shuffle_seed=self.config.random_seed,
                    py_start_method="spawn",
                    **pipeline_kwargs,
                )
                pipeline.build()

                return dali_pipeline.SemSegPipeloader(dataset=dataset, pipeline=pipeline)

            else:
                raise NotImplementedError(
                    f"DALI pipeline can not be used for {dataset.__class__.__name__}"
                )

        else:
            if num_samples is not None:
                indices = [int(i) for i in torch.randperm(len(dataset))[:num_samples]]
                dataset = Subset(dataset, indices)

            return DataLoader(
                dataset=dataset,
                num_workers=num_workers,
                batch_size=batch_size,
                sampler=DistributedSampler(dataset),
                pin_memory=True,
            )
