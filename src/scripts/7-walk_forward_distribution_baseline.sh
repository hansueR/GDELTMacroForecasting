#!/bin/bash
#SBATCH --job-name=WF_DIST_BASE
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --partition=bigTiger
#SBATCH --nodelist=itiger04
#SBATCH --output=/project/hrao/GDELT/logs_AI/%x_%j.log

set -eo pipefail

source /project/hrao/miniconda3/bin/activate
conda activate comp7720_knn

ROOT=/project/hrao/GDELT
cd "$ROOT"

mkdir -p "$ROOT/results/distribution_baselines_v2" "$ROOT/logs_AI"

python scripts_AI/7-walk_forward_distribution_baseline.py \
  --model_table "$ROOT/modeling/model_table.parquet" \
  --output_dir "$ROOT/results/distribution_baselines_v2" \
  --models naive,ewma,ridge \
  --first_test_year 2019 \
  --sigma_floor 1e-6 \
  --ewma_alpha 0.06

# sbatch scripts_AI/7-walk_forward_distribution_baseline.sh