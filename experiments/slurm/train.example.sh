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
# More nodes (data parallel, 4 GPUs per node; the 500K-token batch is split
# across all ranks, so steps get faster and GPU-hours stay the same). The node
# count is an sbatch flag, not an env var, and a chain must keep it (train.py
# refuses a resume with a different world size):
#   MODEL=170m VARIANT=atlas-mac sbatch --nodes=4 experiments/slurm/train.sh
#
# Resume from checkpoint:
#   MODEL=170m VARIANT=atlas-mac RESUME=runs/170m-atlas-mac/step-1000 sbatch experiments/slurm/train.sh
#
# With ablation:
#   MODEL=170m VARIANT=atlas-mac ABLATION=no-omega sbatch experiments/slurm/train.sh
#
# Memory-free trunk baseline (run name gets -vanilla, as train.py does):
#   MODEL=170m VARIANT=atlas-mac VANILLA=1 sbatch experiments/slurm/train.sh
#
# One-off memory overrides (space-separated KEY=VALUE, Python literals) and a
# run-name suffix to keep the variants apart:
#   MEMORY_KWARGS="use_sequential_scan=False" RUN_SUFFIX=-assoc sbatch experiments/slurm/train.sh
#
# Chaining past the 72 h QoS cap: every job after the first resumes from the
# newest complete checkpoint of the same run name. Submit the chain up front:
#   j=$(MODEL=170m VARIANT=atlas-mac sbatch --parsable experiments/slurm/train.sh)
#   for i in 1 2 3; do
#     j=$(MODEL=170m VARIANT=atlas-mac RESUME=latest sbatch --parsable --dependency=afterany:$j experiments/slurm/train.sh)
#   done
# (afterany, not afterok: a job killed at the wall limit exits non-zero.)
# Every job in the chain must repeat the SAME env (ABLATION, VANILLA, RUN_SUFFIX,
# MEMORY_KWARGS, PEAK_LR, TOTAL_TOKENS, SAVE_EVERY): the run name decides which checkpoint
# `latest` finds, and train.py refuses a resume whose peak LR or memory
# overrides differ from the checkpoint's meta.pt.
#
# This file is a template: copy it to experiments/slurm/train.sh (gitignored,
# holds the real account/paths) and re-copy after every update — an old
# train.sh does not know RESUME=latest.

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

# Multi-GPU: one accelerate launcher per node (srun below), 4 processes each,
# rendezvous on the first node of the allocation
export MASTER_ADDR=$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n 1)
export MASTER_PORT=$((29500 + SLURM_JOB_ID % 1000))
NUM_MACHINES=${SLURM_NNODES:-1}
NUM_PROCESSES=$((4 * NUM_MACHINES))
export NCCL_DEBUG=WARN

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
if [ "${RESUME}" = "latest" ]; then
    RESUME_FLAG="--resume latest"          # train.py resolves the newest complete step-* of this run
elif [ -n "${RESUME}" ]; then
    RESUME_FLAG="--resume ${PROJECT_ROOT}/${RESUME}"
fi
VANILLA_FLAG=""
VANILLA_SUFFIX=""
if [ "${VANILLA}" = "1" ]; then        # exactly 1: VANILLA=0 must mean off
    VANILLA_FLAG="--vanilla"
    VANILLA_SUFFIX="-vanilla"
elif [ -n "${VANILLA}" ]; then
    echo "VANILLA must be 1 or unset, got '${VANILLA}'" >&2; exit 2
fi
MEMORY_FLAGS=""
set -f                                  # no glob expansion of the values
for kv in ${MEMORY_KWARGS}; do
    case "${kv}" in
        *[\$\`\*\?\'\"]*|*=|=*) echo "MEMORY_KWARGS entry '${kv}' rejected: KEY=VALUE, no quotes/globs/shell chars (it is re-parsed by the container shell)" >&2; exit 2 ;;
    esac
    MEMORY_FLAGS="${MEMORY_FLAGS} --memory-kwarg ${kv}"
done
set +f
MAX_STEPS_FLAG=""
if [ -n "${MAX_STEPS}" ]; then
    MAX_STEPS_FLAG="--max-steps ${MAX_STEPS}"
fi
WARMUP_FLAG=""
if [ -n "${WARMUP_STEPS}" ]; then
    WARMUP_FLAG="--warmup-steps ${WARMUP_STEPS}"
fi
TOTAL_TOKENS_FLAG=""
if [ -n "${TOTAL_TOKENS}" ]; then
    TOTAL_TOKENS_FLAG="--total-tokens ${TOTAL_TOKENS}"   # cosine span = budget, e.g. 2e9 (recorded + validated on resume)
fi

# Checkpoints: SAVE_EVERY=100 writes ~300 checkpoints x ~2-3 GB over a full
# 15B run (600-900 GB of GPFS). train.py's --keep-checkpoints N rotates,
# keeping only the newest N — opt-in because evals score historical checkpoints.
RUN_NAME="${MODEL}-${VARIANT}${ABLATION:+-${ABLATION}}${VANILLA_SUFFIX}${RUN_SUFFIX}"

# rename job to match variant (SBATCH --job-name is hardcoded; static directives can't use env vars)
scontrol update jobid=${SLURM_JOB_ID} name=${RUN_NAME} 2>/dev/null || true

cd ${PROJECT_ROOT}
mkdir -p runs

# srun starts one task per node; each task's SLURM_NODEID is its machine rank
# (escaped so the task shell expands it, not this one)
srun --ntasks=${NUM_MACHINES} --ntasks-per-node=1 --cpus-per-task=${SLURM_CPUS_PER_TASK:-80} \
    singularity exec --nv \
    --bind ${PROJECT_ROOT}:${PROJECT_ROOT} \
    --bind ${DATA_DIR}:${DATA_DIR} \
    ${CONTAINER} \
    bash -c "cd ${PROJECT_ROOT} && \
        PYTHONPATH=${PROJECT_ROOT}:\${PYTHONPATH} \
        WANDB_MODE=offline \
        accelerate launch \
            --mixed_precision bf16 \
            --num_machines ${NUM_MACHINES} \
            --num_processes ${NUM_PROCESSES} \
            --machine_rank \${SLURM_NODEID} \
            --main_process_ip ${MASTER_ADDR} \
            --main_process_port ${MASTER_PORT} \
            experiments/train.py \
                --model ${MODEL} \
                --variant ${VARIANT} \
                ${ABLATION_FLAG} \
                ${LR_FLAG} \
                ${RESUME_FLAG} \
                ${MAX_STEPS_FLAG} \
                ${WARMUP_FLAG} \
                ${TOTAL_TOKENS_FLAG} \
                ${VANILLA_FLAG} \
                ${MEMORY_FLAGS} \
                --data-dir ${DATA_DIR} \
                --output-dir ${PROJECT_ROOT}/runs \
                --run-name ${RUN_NAME} \
                --wandb \
                --per-device-batch-size 1 \
                --save-every ${SAVE_EVERY:-100} \
                --validate-every ${VALIDATE_EVERY:-1000} \
                --seq-len 1024 \
                --log-every 10"
