#!/bin/bash
#SBATCH --job-name=prepare_data
#SBATCH --account=eporaif01
#SBATCH --qos=gp_ehpc
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=80
#SBATCH --output=runs/%x_%j.log
#SBATCH --error=runs/%x_%j.log

# Pre-tokenize FineWeb on BSC (CPU-only, no GPU needed).
#
# Prerequisites — download these locally and rsync to BSC:
#   1. FineWeb parquet: huggingface-cli download HuggingFaceFW/fineweb sample/10BT --repo-type dataset
#   2. T5 tokenizer:    python -c "from transformers import AutoTokenizer; t = AutoTokenizer.from_pretrained('google-t5/t5-base'); t.save_pretrained('/tmp/t5-tokenizer')"
#
# Then rsync both to GPFS:
#   rsync -avz /tmp/fineweb-parquet/ transfer1.bsc.es:/gpfs/projects/eporaif01/data/fineweb-parquet/
#   rsync -avz /tmp/t5-tokenizer/    transfer1.bsc.es:/gpfs/projects/eporaif01/data/t5-tokenizer/

ml singularity/4.1.5

export PROJECT_ROOT="/gpfs/projects/eporaif01/atlas-torch"
export DATA_ROOT="/gpfs/projects/eporaif01/data"
export CONTAINER="/gpfs/projects/eporaif01/containers/atlas-torch"
export PYTHONUNBUFFERED=1

cd ${PROJECT_ROOT}
mkdir -p runs

singularity exec \
    --bind ${PROJECT_ROOT}:${PROJECT_ROOT} \
    --bind ${DATA_ROOT}:${DATA_ROOT} \
    ${CONTAINER} \
    python experiments/data/prepare.py \
        --output ${DATA_ROOT}/fineweb-t5 \
        --data-dir ${DATA_ROOT}/fineweb-parquet \
        --tokenizer-dir ${DATA_ROOT}/t5-tokenizer \
        --max-tokens 15000000000
