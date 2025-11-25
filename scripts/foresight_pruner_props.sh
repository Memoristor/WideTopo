#!/bin/bash

source source.sh

CONFIG=$1
SEED=$2
PRUNE_GPUS=$3
PRUNE_ARGS=$4

# Get the number of visible GPUS
NUM_PRUNE_GPUS=$(echo "$PRUNE_GPUS" | awk -F',' '{print NF}')

# Get random master port
MASTER_PORT=$(alloc_port 28000 38000)

# Run script
CUDA_VISIBLE_DEVICES="${PRUNE_GPUS}" torchrun \
    --nproc_per_node ${NUM_PRUNE_GPUS} \
    --nnodes 1 \
    --master_port ${MASTER_PORT} \
    pruner_props.py --wandb_project Foresight-Pruner-Props-V2 \
    --yaml_config ${CONFIG} \
    --proc_phase foresight_pruning \
    --random_seed ${SEED} ${PRUNE_ARGS} 
    