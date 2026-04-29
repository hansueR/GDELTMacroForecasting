#!/bin/bash
#SBATCH --job-name=SUM_ABL
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

mkdir -p "$ROOT/results/ablation_tables" "$ROOT/logs_AI"

python scripts_AI/13-summarize_ablation_results.py \
  --event_window_dir "$ROOT/results/event_window_ablation" \
  --topk_root_dir "$ROOT/results/topk_ablation" \
  --output_dir "$ROOT/results/ablation_tables"

# sbatch scripts_AI/13-summarize_ablation_results.sh
