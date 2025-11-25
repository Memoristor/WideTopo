# coding=utf-8

import fnmatch
import os
import random
import socket

import numpy as np
import torch
import wandb
from omegaconf import DictConfig, OmegaConf
from scipy.sparse.linalg import LinearOperator, eigsh
from torch import distributed as dist
from torch.backends import cudnn
from torch.nn.modules.batchnorm import _BatchNorm
from torch.nn.parallel import DataParallel, DistributedDataParallel

__all__ = [
    # Commonly used
    "find_files",
    "seed_init",
    "named_params",
    "wandb_init",
    # For model pruning
    "lanczos_algorithm",
    "hard_bn_momentum",
    "reset_bn_momentum",
    "reset_bn_status",
]


def find_files(directory, pattern):
    """Find all files that matches the pattern under the directory"""
    for root, dirs, files in os.walk(os.path.expanduser(directory)):
        for basename in files:
            if fnmatch.fnmatch(basename, pattern) or fnmatch.fnmatch(
                os.path.join(root, basename), pattern
            ):
                filename = os.path.join(root, basename)
                yield filename


def seed_init(seed=0, cuda_deterministic=True):
    """Init seed of numpy, random, torch"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if cuda_deterministic:  # slower, more reproducible
        cudnn.deterministic = True
        cudnn.benchmark = False
    else:  # faster, less reproducible
        cudnn.deterministic = False
        cudnn.benchmark = True


def named_params(model):
    """Get named parameters of model"""
    if isinstance(model, DataParallel) or isinstance(model, DistributedDataParallel):
        return dict([(n, w) for (n, w) in model.module.named_parameters()])
    else:
        return dict([(n, w) for (n, w) in model.named_parameters()])


def wandb_init(wandb_path, wandb_config: DictConfig, project_name: str, experiment_name: str):
    """Init weight&bias. If the previous run does not exist, a new run will be created.
    Otherwise, the existing run will be loaded."""
    wandb_config = OmegaConf.to_container(wandb_config, resolve=True)
    runid_txt = os.path.join(wandb_path, "run_id.txt")
    if os.path.exists(runid_txt):
        with open(runid_txt, "r") as f:
            run_id = f.read().strip()
        run = wandb.init(
            config=wandb_config,
            project=project_name,
            notes=socket.gethostname(),
            name=experiment_name,
            dir=wandb_path,
            id=run_id,
            resume="allow",
        )
    else:
        run = wandb.init(
            config=wandb_config,
            project=project_name,
            notes=socket.gethostname(),
            name=experiment_name,
            dir=wandb_path,
            resume="auto",
        )
        with open(os.path.join(wandb_path, "run_id.txt"), "w") as f:
            f.write(run.id)


def lanczos_algorithm(mvp_func, dim: int, neigs: int, ncv=None, *args, **kwargs):
    """Invoke the Lanczos algorithm to compute the leading eigenvalues and eigenvectors of a matrix / linear operator
    (which we can access via matrix-vector products).

    Modified from repo: https://github.com/locuslab/edge-of-stability/blob/github/src/utilities.py
    """
    device = torch.device(dist.get_rank())

    def mv(vec: np.ndarray):
        gpu_vec = torch.tensor(vec, dtype=torch.float).to(device)
        return mvp_func(gpu_vec).detach().cpu().numpy()  # matrix vector product function

    operator = LinearOperator((dim, dim), matvec=mv)
    evals, evecs = eigsh(operator, neigs, ncv=ncv, *args, **kwargs)
    evals = torch.from_numpy(np.ascontiguousarray(evals[::-1]).copy()).float().to(device)
    evecs = torch.from_numpy(np.ascontiguousarray(np.flip(evecs, -1)).copy()).float().to(device)
    return evals, evecs


@torch.no_grad()
def hard_bn_momentum(model):
    """Disable momentum of BatchNorm"""

    def func(m):
        if isinstance(m, _BatchNorm):
            m.backup_momentum = m.momentum
            m.momentum = 1.0

    model.apply(func)


@torch.no_grad()
def reset_bn_momentum(model):
    """Disable momentum of BatchNorm"""

    def func(m):
        if isinstance(m, _BatchNorm) and hasattr(m, "backup_momentum"):
            m.momentum = m.backup_momentum

    model.apply(func)


@torch.no_grad()
def reset_bn_status(model):
    """Reset running status of BatchNorm"""

    def func(m):
        if isinstance(m, _BatchNorm):
            m.reset_running_stats()

    model.apply(func)
