#!/bin/bash
#SBATCH --job-name=BUILD_SEQ
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --partition=bigTiger
#SBATCH --nodelist=itiger04
#SBATCH --output=/project/hrao/GDELT/logs_AI/%x_%j.log

set -eo pipefail

source /project/hrao/miniconda3/bin/activate
conda activate comp7720_knn

ROOT=/project/hrao/GDELT
cd "$ROOT"

mkdir -p "$ROOT/modeling/sequence_dataset_l20" "$ROOT/logs_AI"

python scripts_AI/9-build_sequence_dataset.py \
  --model_table "$ROOT/modeling/model_table.parquet" \
  --output_dir "$ROOT/modeling/sequence_dataset_l20" \
  --lookback 20

# sbatch scripts_AI/9-build_sequence_dataset.sh