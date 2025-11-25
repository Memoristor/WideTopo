# coding=utf-8

import json
import os

import numpy as np
import torch
import wandb
from omegaconf import DictConfig
from torch import distributed as dist
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

import losses
import metrics
from datasets import mixup
from engines import BasicEngine
from tools import utils

__all__ = [
    "BasicTrain",
    "ClassificationTrain",
    "SegmentationTrain",
]


class BasicTrain(BasicEngine):
    """The engine is used in the training phase."""

    def __init__(self, *args, **kwags):
        super(BasicTrain, self).__init__(*args, **kwags)

        # Init metrics
        self.metrics = dict()
        if isinstance(self.config.train.metrics, (dict, DictConfig)):
            for k, v in self.config.train.metrics.items():
                self.metrics[k] = getattr(metrics, v.cls)(**v.get("kwargs", {}))
                self.metrics[k].to(torch.device(dist.get_rank()))
                self.metrics[k].reset()

        # Load criterion
        self.criteria = dict()
        for k, v in self.config.train.losses.items():  # train losses
            self.criteria[k] = getattr(losses, v.cls)(**v.get("kwargs", {}))

        # Create mixup/cutmix
        self.mixup = None
        if "mixup" in self.config.train.keys():
            self.mixup = getattr(mixup, self.config.train.mixup.cls)(
                **self.config.train.mixup.get("kwargs", {})
            )

    def losses(self, pred: dict, truth: dict):
        """Calculate the losses of each prediction and corresponding ground truth.

        Params:
            pred (dict): The predict data, e.g. {'body': body, 'edge': edge}
            truth (dict): The truth data, e.g. {'body': body, 'edge': edge}

        Returns:
            return the dict of losses, e.g. {'body': body_loss, 'edge': edge_loss}
        """
        losses = dict()
        for k, v in self.config.train.losses.items():  # train losses
            tensors = list()
            for pairs in v.forward:
                for pk, pv in pairs.items():
                    if pk == "inputs":
                        tensors.append(truth[pv])
                    elif pk == "outputs":
                        tensors.append(pred[pv])
                    else:
                        raise AttributeError(f"Unknown keys: {pk}, not in `inputs` or `outputs`")
            losses[k] = self.criteria[k](*tensors)
        return losses

    def get_wandb_info(self, phase, iters, **kwargs):
        """Get wandb information"""
        wandb.define_metric(f"iteration/{phase}")

        wandb_info = dict()
        wandb_info = {f"iteration/{phase}": iters}
        return wandb_info

    def gradient_backward(self, loss_all):
        """Gradient backward"""
        self.optimizer.zero_grad()
        # loss_all.backward()
        self.grad_scaler.scale(loss_all).backward()

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

    def predict(self, model, truth):
        """Get model predictions

        Params:
            truth (dict). Ground truth.
        """
        input_tensors = list()
        for k in self.config.model.input_keys:
            input_tensors.append(truth[k])

        output_tensors = model(*input_tensors)
        if isinstance(output_tensors, (list, tuple)):
            pred = dict([(k, v) for k, v in zip(self.config.model.output_keys, output_tensors)])
        else:
            pred = {self.config.model.output_keys[0]: output_tensors}
        return pred

    def update_metrics(self, pred, truth):
        """Update metrics

        Params:
            pred (dict): The predict data, e.g. {'body': body, 'edge': edge}
            truth (dict): The truth data, e.g. {'body': body, 'edge': edge}
        """
        for k, v in self.metrics.items():
            update_tensors = list()
            for pairs in self.config.train.metrics[k].tensor_keys:
                for pk, pv in pairs.items():
                    if pk == "inputs":
                        update_tensors.append(truth[pv])
                    elif pk == "outputs":
                        update_tensors.append(pred[pv])
                    else:
                        raise AttributeError("Unknown keys, not in `inputs` or `outputs`")
            v.update(*update_tensors)

    def run_epoch(self, dataloader, epoch, phase="train", mixup=None, summary_freq=10):
        """Run one epoch, and get the metrics and losses.

        Params:
            dataloader (DataLoader): The data loader for the dataset.
            epoch (int): The current epoch number.
            phase (str): The phase of training, options are `train` or `valid`.
            mixup (callable): A callable function or object that applies Mixup or CutMix augmentation.
            summary_freq (int): Frequency of logging summaries. Defaults to 10.
        """
        if phase == "train":
            self.model.train()
            grad_func = torch.enable_grad()
            # Set epoch for distributed sampler
            if isinstance(dataloader, DataLoader):
                dataloader.sampler.set_epoch(epoch)
        elif phase == "valid":
            self.model.eval()
            grad_func = torch.no_grad()
        else:
            raise AttributeError(f"Unknown phase: {phase}, not in `train`, `valid`")

        # Reset metrics
        for v in self.metrics.values():
            v.reset()

        # Run epochs
        epoch_losses = dict()
        epoch_metrics = dict()
        with grad_func:
            bar = tqdm(dataloader)
            for i, truth in enumerate(bar):
                # Epoch start from `1`
                iterations = i + (epoch - 1) * len(bar)

                for k, v in truth.items():
                    truth[k] = v.to(torch.device(dist.get_rank()))

                # Mixup/Cutmix
                if phase == "train" and mixup is not None:
                    assert {
                        "image",
                        "label",
                    } <= truth.keys(), "The key `image` and `label` must in the `truth`"
                    image = truth["image"]
                    label = truth["label"]
                    image, target = mixup(image, label)
                    truth["target"] = target.to(torch.device(dist.get_rank()))
                    truth["label"] = torch.argmax(target, dim=1)

                # Get predict results and losses, update metrics
                if phase == "train":
                    with autocast(enabled=self.grad_scaler.is_enabled()):
                        pred = self.predict(model=self.model, truth=truth)
                        losses = self.losses(pred=pred, truth=truth)
                    loss_all = torch.sum(torch.stack(list(losses.values())))

                    # Gradient backward
                    self.gradient_backward(loss_all=loss_all)

                    # Update learning rate scheduler by `iteration`
                    if self.config.learning_rate.step_mode == "iteration":
                        self.lr_scheduler.step(iterations)
                else:
                    pred = self.predict(model=self.model, truth=truth)
                    losses = self.losses(pred=pred, truth=truth)
                    loss_all = torch.sum(torch.stack(list(losses.values())))

                self.update_metrics(pred=pred, truth=truth)

                # flush wandb summary
                if (
                    dist.get_rank() == 0
                    and not self.config.disable_wandb
                    and iterations % summary_freq == 0
                ):
                    wandb_info = self.get_wandb_info(
                        phase,
                        iterations,
                        # pred=pred,
                        # truth=truth,
                        # dataloader=dataloader,
                        # epoch=epoch,
                    )
                    wandb.log(wandb_info)

                # Set bar description
                last_lr = self.lr_scheduler.get_lr()[0]
                bar.set_description(
                    f"[{phase}] Epoch: {epoch}/{self.config.train.num_epoch}, Rank: {dist.get_rank()}, Loss: {loss_all:5.5f}, LR: {last_lr:5.5f}"
                )

                # Reduces the loss data across all machines
                for k, v in losses.items():
                    dist.all_reduce(v, op=dist.ReduceOp.SUM)
                    v.div_(dist.get_world_size())
                    if k not in epoch_losses.keys():
                        epoch_losses[k] = v.detach().cpu().numpy() / len(dataloader)
                    else:
                        epoch_losses[k] += v.detach().cpu().numpy() / len(dataloader)

            # Get metric of this epoch
            for k, v in self.metrics.items():
                epoch_metrics[k] = v.compute().cpu().numpy()

        # Update learning rate scheduler by `epoch`
        if phase == "train" and self.config.learning_rate.step_mode == "epoch":
            self.lr_scheduler.step(epoch)

        return epoch_losses, epoch_metrics

    def run(self):
        """Train model"""
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
            wandb.define_metric("losses/*", step_metric="epoch")
            wandb.define_metric("metrics/*", step_metric="epoch")
            wandb.define_metric("learning_rate/*", step_metric="epoch")

        # Load the best validation metric records
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

            # Run epoch
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

            # Save and record results
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


class ClassificationTrain(BasicTrain):
    """The engine is used in the classification training phase."""

    def get_wandb_info(self, phase, iters, **kwargs):
        """Get wandb information"""
        # Set parameters of wandb
        wandb.define_metric(f"iteration/{phase}")
        wandb.define_metric("table/*", step_metric=f"iteration/{phase}")

        wandb_info = dict()
        wandb_info = {f"iteration/{phase}": iters}

        if {"truth", "pred", "dataloader", "epoch"} <= kwargs.keys():
            truth = kwargs["truth"]
            pred = kwargs["pred"]
            dataloader = kwargs["dataloader"]
            epoch = kwargs["epoch"]

            if {"image", "label"} <= truth.keys() and "logit" in pred.keys():
                index = np.random.randint(0, truth["image"].shape[0])
                image = truth["image"][index].cpu().numpy().transpose((1, 2, 0))
                image_std = dataloader.dataset.image_std
                image_mean = dataloader.dataset.image_mean
                if dataloader.dataset.div_std is True:
                    image = image * image_std + image_mean
                else:
                    image = image + image_mean
                image = np.asarray(image * 255, dtype=np.uint8)

                label = truth["label"][index].cpu().numpy()
                label = dataloader.dataset.CLASSES[int(label)]

                softmax = torch.softmax(pred["logit"][index], dim=0)
                argmax = torch.argmax(softmax, dim=0).cpu().detach().numpy()
                result = dataloader.dataset.CLASSES[argmax]
                accuracy = softmax[argmax]

                columns = ["Image", "Label", "Result", "Accuracy"]
                data = [
                    [
                        wandb.Image(image, caption=f"epoch_{epoch}_iters_{iters}"),
                        label,
                        result,
                        accuracy,
                    ],
                ]

                table = wandb.Table(data=data, columns=columns)
                wandb_info[f"table/{phase}"] = table

        return wandb_info


class SegmentationTrain(BasicTrain):
    """The engine is used in the segmentation training phase."""

    def get_wandb_info(self, phase, iters, **kwargs):
        """Get wandb information"""
        # Set parameters of wandb
        wandb.define_metric(f"iteration/{phase}")
        wandb.define_metric("table/*", step_metric=f"iteration/{phase}")

        wandb_info = dict()
        wandb_info = {f"iteration/{phase}": iters}

        if {"truth", "pred", "dataloader", "epoch"} <= kwargs.keys():
            truth = kwargs["truth"]
            pred = kwargs["pred"]
            dataloader = kwargs["dataloader"]
            epoch = kwargs["epoch"]

            if {"image", "label"} <= truth.keys() and "logit" in pred.keys():
                index = np.random.randint(0, truth["image"].shape[0])

                image = truth["image"][index].cpu().numpy().transpose((1, 2, 0))
                image_std = dataloader.dataset.image_std
                image_mean = dataloader.dataset.image_mean
                if dataloader.dataset.div_std is True:
                    image = image * image_std + image_mean
                else:
                    image = image + image_mean
                image = np.asarray(image * 255, dtype=np.uint8)

                label = truth["label"][index].cpu().numpy()
                label = dataloader.dataset.decode_label(label)

                result = torch.argmax(pred["logit"][index], dim=0)
                result = result.cpu().detach().numpy()
                result = dataloader.dataset.decode_label(result)

                columns = ["Image", "Label", "Result"]
                data = [
                    [
                        wandb.Image(image, caption=f"epoch_{epoch}_iters_{iters}"),
                        wandb.Image(label, caption=f"epoch_{epoch}_iters_{iters}"),
                        wandb.Image(result, caption=f"epoch_{epoch}_iters_{iters}"),
                    ],
                ]

            table = wandb.Table(data=data, columns=columns)
            wandb_info[f"table/{phase}"] = table

        return wandb_info
