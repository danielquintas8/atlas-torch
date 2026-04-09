#!/bin/bash
#SBATCH --job-name=atlas_train
#SBATCH --account=eporaif01
#SBATCH --qos=acc_ehpc
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=80
#SBATCH --output=runs/%x_%j.log
#SBATCH --error=runs/%x_%j.log

# Usage:
#   MODEL=170m VARIANT=atlas-mac sbatch experiments/slurm/train.sh
#
# With ablation:
#   MODEL=170m VARIANT=atlas-mac ABLATION=no-omega sbatch experiments/slurm/train.sh

ml singularity/4.1.5

export PROJECT_ROOT="/gpfs/projects/eporaif01/atlas-torch"
export DATA_DIR="/gpfs/projects/eporaif01/data/fineweb-t5"
export CONTAINER="/gpfs/projects/eporaif01/containers/atlas-torch"
export PYTHONUNBUFFERED=1

# CUDA/PyTorch
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=20

# Override Intel compilers
export CC=gcc
export CXX=g++

# Multi-GPU (single node)
export MASTER_ADDR=localhost
export MASTER_PORT=$((29500 + SLURM_JOB_ID % 1000))

# Defaults
MODEL=${MODEL:-170m}
VARIANT=${VARIANT:-atlas-mac}
ABLATION_FLAG=""
if [ -n "${ABLATION}" ]; then
    ABLATION_FLAG="--ablation ${ABLATION}"
fi

RUN_NAME="${MODEL}-${VARIANT}${ABLATION:+-${ABLATION}}"

cd ${PROJECT_ROOT}
mkdir -p runs

singularity exec --nv \
    --bind ${PROJECT_ROOT}:${PROJECT_ROOT} \
    --bind ${DATA_DIR}:${DATA_DIR} \
    ${CONTAINER} \
    bash -c "cd ${PROJECT_ROOT} && \
        PYTHONPATH=${PROJECT_ROOT}:\${PYTHONPATH} \
        WANDB_MODE=offline \
        accelerate launch \
            --mixed_precision bf16 \
            --num_machines 1 \
            --num_processes 4 \
            --main_process_ip localhost \
            --main_process_port ${MASTER_PORT} \
            experiments/train.py \
                --model ${MODEL} \
                --variant ${VARIANT} \
                ${ABLATION_FLAG} \
                --data-dir ${DATA_DIR} \
                --output-dir ${PROJECT_ROOT}/runs \
                --run-name ${RUN_NAME} \
                --wandb \
                --per-device-batch-size 4 \
                --save-every 5000 \
                --validate-every 1000 \
                --log-every 10"
