#!/bin/bash
#SBATCH --job-name=ASSET_PRICE
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --partition=bigTiger
#SBATCH --nodelist=itiger04
#SBATCH --output=/project/hrao/GDELT/logs/%x_%j.log

set -eo pipefail

source /project/hrao/miniconda3/bin/activate
conda activate comp7720_knn

ROOT=/project/hrao/GDELT
cd "$ROOT"

mkdir -p "$ROOT/asset_prices" "$ROOT/logs"

pip show yfinance >/dev/null 2>&1 || pip install yfinance

python scripts_AI/3-download_asset_prices.py \
  --start 2016-03-01 \
  --end 2026-03-31 \
  --output_file "$ROOT/asset_prices/daily_asset_prices.parquet" \
  --tickers "QQQ=QQQ,Gold=GLD,WTI_Oil=USO"


# sbatch scripts/3-download_asset_prices.sh
