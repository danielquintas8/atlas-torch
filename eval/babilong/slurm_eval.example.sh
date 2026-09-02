#!/bin/bash
#SBATCH --job-name=atlas_eval
#SBATCH --account=YOUR_BSC_ACCOUNT
#SBATCH --qos=acc_debug
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --output=runs/%x_%j.log
#SBATCH --error=runs/%x_%j.log

# Usage:
#   CKPT=runs/170m-titans-mac/step-4000 VARIANT=titans-mac sbatch eval/babilong/slurm_eval.example.sh
#
# Override the default smoke-test grid:
#   CKPT=... VARIANT=... LENGTHS="4k 16k 64k" TASKS="qa1 qa2 qa3" MAX_EXAMPLES=100 sbatch ...

ml singularity/4.1.5

export PROJECT_ROOT="/gpfs/projects/YOUR_BSC_ACCOUNT/atlas-torch"
export DATA_DIR="/gpfs/projects/YOUR_BSC_ACCOUNT/data/fineweb-t5"
export CONTAINER="/gpfs/projects/YOUR_BSC_ACCOUNT/containers/atlas-torch"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Offline HF: dataset + tokenizer + hub entirely from local cache
export HF_HOME="${PROJECT_ROOT}/hf_cache"
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

export CC=gcc
export CXX=g++

# Inputs (required)
CKPT=${CKPT:?"set CKPT=runs/<run-name>/step-XXXX"}
VARIANT=${VARIANT:?"set VARIANT=titans-mac|atlas-mac|titans-mag|atlas-mag"}
MODEL=${MODEL:-170m}
ABLATION=${ABLATION:-}

# Eval grid (smoke-test default: tiny)
LENGTHS=${LENGTHS:-"4k"}
TASKS=${TASKS:-"qa1"}
MAX_EXAMPLES=${MAX_EXAMPLES:-10}

ABLATION_FLAG=""
if [ -n "${ABLATION}" ]; then
    ABLATION_FLAG="--ablation ${ABLATION}"
fi

# Vanilla baseline (Ablation 4): build memory-free model to match the checkpoint.
VANILLA_FLAG=""
if [ "${VANILLA}" = "1" ]; then
    VANILLA_FLAG="--vanilla"
fi

RUN_NAME="${MODEL}-${VARIANT}${ABLATION:+-${ABLATION}}"
OUTPUT="${PROJECT_ROOT}/runs/eval-${RUN_NAME}-${SLURM_JOB_ID}.json"

cd ${PROJECT_ROOT}
mkdir -p runs

singularity exec --nv \
    --bind ${PROJECT_ROOT}:${PROJECT_ROOT} \
    --bind ${DATA_DIR}:${DATA_DIR} \
    ${CONTAINER} \
    bash -c "cd ${PROJECT_ROOT} && \
        PYTHONPATH=${PROJECT_ROOT}:\${PYTHONPATH} \
        python eval/babilong/evaluate.py \
            --checkpoint ${PROJECT_ROOT}/${CKPT} \
            --model ${MODEL} \
            --variant ${VARIANT} \
            ${ABLATION_FLAG} \
            ${VANILLA_FLAG} \
            --tokenizer-dir ${DATA_DIR}/tokenizer \
            --lengths ${LENGTHS} \
            --tasks ${TASKS} \
            --max-examples ${MAX_EXAMPLES} \
            --output ${OUTPUT}"
