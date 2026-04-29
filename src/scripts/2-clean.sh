#!/bin/bash
#SBATCH --job-name=GDELT_CLEAN
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
RAW_DIR="$ROOT/raw_zip"
CLEAN_DIR="$ROOT/clean"
LOG_DIR="$ROOT/logs"
SCRIPT="$ROOT/scripts_AI/2-clean_gdelt_month.py"

mkdir -p "$CLEAN_DIR" "$LOG_DIR"
cd "$ROOT"

shopt -s nullglob

for monthdir in "$RAW_DIR"/*; do
    [[ -d "$monthdir" ]] || continue

    month=$(basename "$monthdir")
    outdir="$CLEAN_DIR/events_$month"
    doneflag="$LOG_DIR/clean_${month}.done"
    logfile="$LOG_DIR/clean_${month}.log"

    if [[ -f "$doneflag" ]]; then
        echo "[$(date)] skip $month (already cleaned)"
        continue
    fi

    mkdir -p "$outdir"

    echo "[$(date)] start clean $month" | tee -a "$logfile"
    echo "input:  $monthdir" | tee -a "$logfile"
    echo "output: $outdir" | tee -a "$logfile"

    set +e
    python "$SCRIPT" \
      --input_dir "$monthdir" \
      --output_dir "$outdir" >> "$logfile" 2>&1
    clean_status=$?
    set -e

    got=$(find "$outdir" -maxdepth 1 -type f -name '*.parquet' | wc -l)

    echo "clean_status=$clean_status parquet_days=$got" | tee -a "$logfile"

    if [[ "$clean_status" -eq 0 && "$got" -gt 0 ]]; then
        touch "$doneflag"
        echo "[$(date)] done clean $month" | tee -a "$logfile"
    else
        echo "[$(date)] WARNING: clean failed for $month" | tee -a "$logfile"
    fi
done

echo "[$(date)] all available months cleaned."

# sbatch scripts/2-clean.sh
