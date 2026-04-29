#!/bin/bash
#SBATCH --job-name=SUM_EVAL
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --partition=bigTiger
#SBATCH --nodelist=itiger04
#SBATCH --output=/project/hrao/GDELT/logs_AI/%x_%j.log

set -eo pipefail

source /project/hrao/miniconda3/bin/activate
conda activate comp7720_knn

ROOT=/project/hrao/GDELT
cd "$ROOT"

mkdir -p "$ROOT/results/module4_evaluation_tables" "$ROOT/logs_AI"

python scripts_AI/8-summarize_evaluation_metrics.py \
  --distribution_by_year "$ROOT/results/distribution_baselines_v2/distribution_baselines_by_year.csv" \
  --output_dir "$ROOT/results/module4_evaluation_tables"

# sbatch scripts_AI/8-summarize_evaluation_metrics.sh