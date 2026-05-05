#!/bin/bash
#SBATCH --job-name=atlas_smoke
#SBATCH --account=eporaif01
#SBATCH --qos=acc_debug
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --output=runs/%x_%j.log
#SBATCH --error=runs/%x_%j.log

# Phase 0 smoke training — 30-min validation that the paper-faithful Atlas
# path (PR #18 + #19 merged) trains without OOM, NaN, or shape bugs before
# committing to a 5-day Atlas 8B retrain.
#
# Goals (per phase2-retrain-plan.md follow-up):
#   1. Training memory fits H100 with the asymmetric MLP path (poly_project_back=False)
#   2. val_loss drops smoothly in first ~50 steps
#   3. Newton-Schulz fires on matrix surprises (regression-tested in test_atlas_muon_*;
#      this run also prints peak GPU memory at the end)
#   4. Per-token retrieve memory is bounded (~hundreds of MB at seq_len=1024)
#
# Usage:
#   VARIANT=atlas-mac sbatch experiments/slurm/smoke.sh
#   VARIANT=titans-mac sbatch experiments/slurm/smoke.sh   # baseline reference
#
# Default: 200 steps × 0.5M tokens/step = 100M tokens, ~30 min wall on 1x H100.
# Single GPU for cleaner memory accounting (4-GPU DDP changes the per-rank budget).

ml singularity/4.1.5

export PROJECT_ROOT="/gpfs/projects/eporaif01/atlas-torch"
export DATA_DIR="/gpfs/projects/eporaif01/data/fineweb-t5"
export CONTAINER="/gpfs/projects/eporaif01/containers/atlas-torch"
export PYTHONUNBUFFERED=1

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=20

export CC=gcc
export CXX=g++

export MASTER_ADDR=localhost
export MASTER_PORT=$((29500 + SLURM_JOB_ID % 1000))

# Defaults — overridable via env
MODEL=${MODEL:-170m}
VARIANT=${VARIANT:-atlas-mac}
MAX_STEPS=${MAX_STEPS:-200}    # 200 * 0.5M tok/step = 100M tokens
SEQ_LEN=${SEQ_LEN:-1024}

# Optional ablation: no-muon, no-poly, no-omega (passes through to train.py).
# Useful for runtime sanity ablations alongside the main smoke — e.g. submit
# both VARIANT=atlas-mac and VARIANT=atlas-mac ABLATION=no-muon, then compare
# their loss curves to verify Muon meaningfully changes the dynamics (not just
# fires per the regression test).
ABLATION_FLAG=""
if [ -n "${ABLATION}" ]; then
    ABLATION_FLAG="--ablation ${ABLATION}"
fi

RUN_NAME="${MODEL}-${VARIANT}${ABLATION:+-${ABLATION}}-smoke"

scontrol update jobid=${SLURM_JOB_ID} name=${RUN_NAME} 2>/dev/null || true

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
            --num_processes 1 \
            --main_process_ip localhost \
            --main_process_port ${MASTER_PORT} \
            experiments/train.py \
                --model ${MODEL} \
                ${ABLATION_FLAG} \
                --variant ${VARIANT} \
                --data-dir ${DATA_DIR} \
                --output-dir ${PROJECT_ROOT}/runs \
                --run-name ${RUN_NAME} \
                --per-device-batch-size 1 \
                --max-steps ${MAX_STEPS} \
                --warmup-steps 50 \
                --save-every 10000 \
                --validate-every 50 \
                --seq-len ${SEQ_LEN} \
                --log-every 10"

# train.py prints PEAK_GPU_MEM_GB at the end of training. Grep the log to
# extract it after the run completes:
#   grep PEAK_GPU_MEM_GB runs/${RUN_NAME}_${SLURM_JOB_ID}.log
