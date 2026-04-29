#!/bin/bash
#SBATCH --job-name=REPORT_TABLES
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

mkdir -p "$ROOT/results/report_summary_tables" "$ROOT/logs_AI"

python scripts_AI/14-build_report_summary_tables.py \
  --root "$ROOT" \
  --output_dir "$ROOT/results/report_summary_tables"

# sbatch scripts_AI/14-build_report_summary_tables.sh
