# coding=utf-8

import argparse
import os
import warnings

import torch
from omegaconf import OmegaConf
from torch import distributed as dist

import engines
from tools import seed_init

warnings.filterwarnings("ignore")


def parse_args():
    parser = argparse.ArgumentParser(description="Experiment entrance")
    parser.add_argument(
        "-c",
        "--yaml_config",
        type=str,
        help="Yaml configuration file required for the experiment",
    )
    parser.add_argument(
        "-p",
        "--proc_phase",
        type=str,
        default="train",
        help="The current phase of the program",
    )
    parser.add_argument(
        "-s",
        "--convert_syncbn",
        action="store_true",
        help="Whether convert BatchNorm to SyncBatchNorm",
    )
    parser.add_argument(
        "-w",
        "--wandb_project",
        type=str,
        default="Foresight-Pruner",
        help="Project name for wandb",
    )
    parser.add_argument(
        "-r",
        "--random_seed",
        default=2024,
        type=int,
        help="Random seed for numpy, torch, random",
    )
    parser.add_argument(
        "-i",
        "--disable_wandb",
        action="store_true",
        help="Do not use wandb to record experimental data",
    )
    parser.add_argument(
        "-d",
        "--delete_history",
        action="store_true",
        help="Whether delete the historical experimental data of the input configuration",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    config = OmegaConf.load(args.yaml_config)

    # Get local rank for initialization
    local_rank = int(os.getenv("LOCAL_RANK", -1))

    # Set wandb url and api key
    os.environ["WANDB_BASE_URL"] = "http://<your_wandb_ip>:8960"
    os.environ["WANDB_API_KEY"] = "local-xxx"

    # Set cuda device to avoid unblanced distributed data parallel
    torch.cuda.set_device(local_rank)

    # Init seeds and process group
    dist.init_process_group(backend="nccl")
    seed_init(config.random_seed + local_rank, cuda_deterministic=(args.proc_phase == "train"))

    # Run engine
    experiment_name = ".".join(os.path.basename(args.yaml_config).split(".")[:-1])
    experiment_name += f"_seed_{config.random_seed}"
    getattr(engines, "ModelTrainingDynamics")(
        project_name=args.wandb_project,
        experiment_name=experiment_name,
        config=config,
        phase=args.proc_phase,
        delete_history=args.delete_history,
    ).run()

    # Cleanup
    dist.destroy_process_group()
