#!/bin/bash
#SBATCH --job-name=TOPK_ABL
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=24:00:00
#SBATCH --partition=bigTiger
#SBATCH --nodelist=itiger04
#SBATCH --output=/project/hrao/GDELT/logs_AI/%x_%j.log

set -eo pipefail

source /project/hrao/miniconda3/bin/activate
conda activate comp7720_knn

ROOT=/project/hrao/GDELT
cd "$ROOT"

mkdir -p "$ROOT/modeling" "$ROOT/results/topk_ablation" "$ROOT/logs_AI"

declare -A EVENT_COL_COUNTS

for K in 10 20 50; do
    echo "Running top-k ablation K=$K"

    MODEL_DIR="$ROOT/modeling/topk_${K}"
    RESULT_DIR="$ROOT/results/topk_ablation/topk_${K}"

    mkdir -p "$MODEL_DIR" "$RESULT_DIR"

    echo "Building model table for K=$K"
    python scripts_AI/5-build_model_table.py \
      --price_file "$ROOT/asset_prices/daily_asset_prices.parquet" \
      --event_long_dir "$ROOT/features/event_long" \
      --output_file "$MODEL_DIR/model_table.parquet" \
      --top_root_k "$K" \
      --top_country_k "$K"

    EVENT_COL_COUNTS[$K]=$(python - "$MODEL_DIR/model_table.parquet" <<'PY'
import sys
import pandas as pd

df = pd.read_parquet(sys.argv[1])
print(sum(c.startswith("event_") for c in df.columns))
PY
)
    echo "K=$K event_column_count=${EVENT_COL_COUNTS[$K]}"

    echo "Running distribution baseline for K=$K"
    python scripts_AI/7-walk_forward_distribution_baseline.py \
      --model_table "$MODEL_DIR/model_table.parquet" \
      --output_dir "$RESULT_DIR" \
      --models ridge \
      --first_test_year 2019 \
      --sigma_floor 1e-6 \
      --ewma_alpha 0.06 \
      --experiment_preset standard
done

if [[ "${EVENT_COL_COUNTS[10]}" -ge "${EVENT_COL_COUNTS[20]}" ]]; then
    echo "ERROR: Expected topk_10 to have fewer event columns than topk_20"
    exit 1
fi

if [[ "${EVENT_COL_COUNTS[20]}" -ge "${EVENT_COL_COUNTS[50]}" ]]; then
    echo "ERROR: Expected topk_20 to have fewer event columns than topk_50"
    exit 1
fi

# sbatch scripts_AI/12-run_topk_ablation.sh
