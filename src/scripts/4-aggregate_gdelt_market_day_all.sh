#!/bin/bash
#SBATCH --job-name=GDELT_AGG
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=50G
#SBATCH --time=12:00:00
#SBATCH --partition=bigTiger
#SBATCH --nodelist=itiger04
#SBATCH --output=/project/hrao/GDELT/logs/%x_%j.log

set -eo pipefail

source /project/hrao/miniconda3/bin/activate
conda activate comp7720_knn

ROOT=/project/hrao/GDELT
CLEAN_DIR="$ROOT/clean"
FEATURE_DIR="$ROOT/features/event_long"
LOG_DIR="$ROOT/logs"

mkdir -p "$FEATURE_DIR" "$LOG_DIR"
cd "$ROOT"

shopt -s nullglob

for monthdir in "$CLEAN_DIR"/events_*; do
    [[ -d "$monthdir" ]] || continue

    month=$(basename "$monthdir" | sed 's/events_//')
    outfile="$FEATURE_DIR/events_market_day_long_${month}.parquet"
    doneflag="$LOG_DIR/agg_${month}.done"

    if [[ -f "$doneflag" && -f "$outfile" ]]; then
        echo "[$(date)] skip $month"
        continue
    fi

    parquet_count=$(find "$monthdir" -maxdepth 1 -type f -name "*.parquet" | wc -l)

    if [[ "$parquet_count" -eq 0 ]]; then
        echo "[$(date)] WARNING: skip $month because no parquet files found in $monthdir"
        echo "$month $monthdir" >> "$LOG_DIR/missing_event_clean_months.txt"
        continue
    fi

    echo "[$(date)] aggregate $month"

    python scripts_AI/4-aggregate_market_day.py \
      --input_dir "$monthdir" \
      --output_file "$outfile" \
      --close_hour 16

    touch "$doneflag"
    echo "[$(date)] done $month"
done

echo "[$(date)] all monthly event aggregation done."
