#!/bin/bash
#SBATCH --job-name=atlas_train
#SBATCH --account=YOUR_BSC_ACCOUNT
#SBATCH --qos=acc_ehpc
#SBATCH --time=72:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=80
#SBATCH --output=runs/%x_%j.log
#SBATCH --error=runs/%x_%j.log

# Usage:
#   MODEL=170m VARIANT=atlas-mac sbatch experiments/slurm/train.sh
#
# Resume from checkpoint:
#   MODEL=170m VARIANT=atlas-mac RESUME=runs/170m-atlas-mac/step-1000 sbatch experiments/slurm/train.sh
#
# With ablation:
#   MODEL=170m VARIANT=atlas-mac ABLATION=no-omega sbatch experiments/slurm/train.sh

ml singularity/4.1.5

export PROJECT_ROOT="/gpfs/projects/YOUR_BSC_ACCOUNT/atlas-torch"
export DATA_DIR="/gpfs/projects/YOUR_BSC_ACCOUNT/data/fineweb-t5"
export CONTAINER="/gpfs/projects/YOUR_BSC_ACCOUNT/containers/atlas-torch"
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
PEAK_LR=${PEAK_LR:-}
ABLATION_FLAG=""
if [ -n "${ABLATION}" ]; then
    ABLATION_FLAG="--ablation ${ABLATION}"
fi
LR_FLAG=""
if [ -n "${PEAK_LR}" ]; then
    LR_FLAG="--peak-lr ${PEAK_LR}"
fi
RESUME_FLAG=""
if [ -n "${RESUME}" ]; then
    RESUME_FLAG="--resume ${PROJECT_ROOT}/${RESUME}"
fi
MAX_STEPS_FLAG=""
if [ -n "${MAX_STEPS}" ]; then
    MAX_STEPS_FLAG="--max-steps ${MAX_STEPS}"
fi
WARMUP_FLAG=""
if [ -n "${WARMUP_STEPS}" ]; then
    WARMUP_FLAG="--warmup-steps ${WARMUP_STEPS}"
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
                ${LR_FLAG} \
                ${RESUME_FLAG} \
                ${MAX_STEPS_FLAG} \
                ${WARMUP_FLAG} \
                --data-dir ${DATA_DIR} \
                --output-dir ${PROJECT_ROOT}/runs \
                --run-name ${RUN_NAME} \
                --wandb \
                --per-device-batch-size 1 \
                --save-every ${SAVE_EVERY:-100} \
                --validate-every 1000 \
                --seq-len 1024 \
                --log-every 10"
