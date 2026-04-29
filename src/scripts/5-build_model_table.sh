#!/bin/bash
#SBATCH --job-name=MODEL_TABLE
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=12:00:00
#SBATCH --partition=bigTiger
#SBATCH --nodelist=itiger04
#SBATCH --output=/project/hrao/GDELT/logs_AI/%x_%j.log

set -eo pipefail

source /project/hrao/miniconda3/bin/activate
conda activate comp7720_knn

ROOT=/project/hrao/GDELT
cd "$ROOT"

mkdir -p "$ROOT/modeling" "$ROOT/logs_AI"

python scripts_AI/5-build_model_table.py \
  --price_file "$ROOT/asset_prices/daily_asset_prices.parquet" \
  --event_long_dir "$ROOT/features/event_long" \
  --output_file "$ROOT/modeling/model_table.parquet" \
  --top_root_k 20 \
  --top_country_k 20

# sbatch scripts/5-build_model_table.sh
