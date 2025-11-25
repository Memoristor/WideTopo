# coding=utf-8

import json
import os

import torch
from omegaconf import DictConfig, OmegaConf
from prettytable import prettytable
from torch import distributed as dist
from tqdm import tqdm

import losses
import metrics
from engines import BasicEngine

__all__ = [
    "BasicValid",
    "ClassificationValid",
    "SegmentationValid",
]


class BasicValid(BasicEngine):
    """The engine is used in the validation phase."""

    def __init__(self, *args, **kwags):
        super(BasicValid, self).__init__(*args, **kwags)

        # Init metrics
        self.metrics = dict()
        if isinstance(self.config[self.phase].metrics, (dict, DictConfig)):
            for k, v in self.config[self.phase].metrics.items():
                self.metrics[k] = getattr(metrics, v.cls)(**v.get("kwargs", {}))
                self.metrics[k].to(torch.device(dist.get_rank()))
                self.metrics[k].reset()

        # Load criteria
        self.criteria = dict()
        for k, v in self.config.train.losses.items():  # train losses
            self.criteria[k] = getattr(losses, v.cls)(**v.get("kwargs", {}))

    def losses(self, pred: dict, truth: dict):
        """Calculate the losses of each prediction and corresponding ground truth.

        Params:
            pred: dict. The predict data, e.g. {'body': body, 'edge': edge}
            truth: dict. The truth data, e.g. {'body': body, 'edge': edge}

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
                        raise AttributeError("Unknown keys, not in `inputs` or `outputs`")
            losses[k] = self.criteria[k](*tensors)
        return losses

    def predict(self, model, truth):
        """Get model predictions

        Params:
            truth. dict. Ground truth.
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
            pred: dict. The predict data, e.g. {'body': body, 'edge': edge}
            truth: dict. The truth data, e.g. {'body': body, 'edge': edge}
        """
        for k, v in self.metrics.items():
            update_tensors = list()
            for pairs in self.config[self.phase].metrics[k].tensor_keys:
                for pk, pv in pairs.items():
                    if pk == "inputs":
                        update_tensors.append(truth[pv])
                    elif pk == "outputs":
                        update_tensors.append(pred[pv])
                    else:
                        raise AttributeError("Unknown keys, not in `inputs` or `outputs`")
            v.update(*update_tensors)

    def run_epoch(self, dataloader):
        """Run one epoch, and get the metrics and losses.

        Params:
            dataloader: DataLoader.
        """
        self.model.eval()

        # Reset metrics
        for v in self.metrics.values():
            v.reset()

        # Run epochs
        valid_losses = dict()
        valid_metrics = dict()
        with torch.no_grad():
            bar = tqdm(dataloader)
            for i, truth in enumerate(bar):
                for k, v in truth.items():
                    truth[k] = v.to(torch.device(dist.get_rank()))

                pred = self.predict(model=self.model, truth=truth)
                losses = self.losses(pred=pred, truth=truth)
                loss_all = torch.sum(torch.stack(list(losses.values())))

                self.update_metrics(pred=pred, truth=truth)

                # Reduces the loss data across all machines
                for k, v in losses.items():
                    dist.all_reduce(v, op=dist.ReduceOp.SUM)
                    v.div_(dist.get_world_size())
                    if k not in valid_losses.keys():
                        valid_losses[k] = v.detach().cpu().numpy() / len(dataloader)
                    else:
                        valid_losses[k] += v.detach().cpu().numpy() / len(dataloader)

                # Set bar description
                bar.set_description(
                    f"[{self.phase}] Rank: {dist.get_rank()}, Loss: {loss_all:5.5f}"
                )

            # Get metric of this epoch
            for k, v in self.metrics.items():
                valid_metrics[k] = v.compute().cpu().numpy()

        return valid_losses, valid_metrics

    def run(self):
        """Valid model"""
        self.model.eval()

        if dist.get_rank() == 0:
            self.logger.info(
                f"[{self.phase}] model: {self.model_class}, dataset: {self.config.dataset.cls}"
            )

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
            os.makedirs(os.path.join(self.save_result, self.phase), exist_ok=True)
            config_file = os.path.join(self.save_result, self.phase, "config.json")
            result_file = os.path.join(self.save_result, self.phase, "result.txt")

            with open(config_file, "w") as f:
                config = OmegaConf.to_container(self.config, resolve=True)
                f.write(json.dumps(config, indent=2))

            self.logger.info(f"[Valid] Configurations have been saved to: {config_file}")

            table_wi_average = prettytable.PrettyTable()
            table_wo_average = prettytable.PrettyTable()
            table_wi_average.field_names = ["", "value"]
            table_wo_average.field_names = ["", *valid_dataloader.dataset.CLASSES]
            for k, metric in self.metrics.items():
                if metric.average == "none" or metric.average is None:
                    table_wo_average.add_row([k, *[f"{x:4.4f}" for x in valid_metrics[k]]])
                else:
                    table_wi_average.add_row([k, f"{float(valid_metrics[k]):4.4f}"])

            with open(result_file, "w") as f:
                f.write(
                    f"[{self.phase}] Model: {self.model_class}, Optimizer: {self.config.optimizer.cls}\n"
                )
                f.write("\n")
                f.write(f"[{self.phase}] The {self.phase} losses\n")
                for n, s in valid_losses.items():
                    f.write(f"{n}: {s:4.4f}\n")
                f.write("\n")
                f.write(f"[{self.phase}] The {self.phase} metric table\n")
                if len(table_wi_average.rows) > 0:
                    f.write(str(table_wi_average))
                if len(table_wo_average.rows) > 0:
                    f.write(str(table_wo_average))
                f.write("\n")

            self.logger.info(f"[{self.phase}] Results have been saved to: {result_file}")

            if len(table_wi_average.rows) > 0:
                self.logger.info(f"{self.phase} metric table:\n{str(table_wi_average)}")
            if len(table_wo_average.rows) > 0:
                self.logger.info(f"{self.phase} metric table:\n{str(table_wo_average)}")


class ClassificationValid(BasicValid):
    """
    The engine is used for the validation phase.
    """


class SegmentationValid(BasicValid):
    """
    The engine is used for the validation phase.
    """
