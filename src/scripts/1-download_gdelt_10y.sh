#!/bin/bash
#SBATCH --job-name=GDELT
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
MANIFEST_DIR="$ROOT/manifests/by_month_201603_202603"
RAW_DIR="$ROOT/raw_zip"
LOG_DIR="$ROOT/logs"
PARALLEL=16

mkdir -p "$RAW_DIR" "$LOG_DIR"
cd "$ROOT"

shopt -s nullglob

for mf in "$MANIFEST_DIR"/*.txt; do
    month=$(basename "$mf" .txt)
    outdir="$RAW_DIR/$month"
    doneflag="$LOG_DIR/download_${month}.done"
    logfile="$LOG_DIR/download_${month}.log"

    mkdir -p "$outdir"

    if [[ -f "$doneflag" ]]; then
        echo "[$(date)] skip $month (already done)"
        continue
    fi

    echo "[$(date)] start $month" | tee -a "$logfile"
    echo "manifest: $mf" | tee -a "$logfile"
    echo "outdir:   $outdir" | tee -a "$logfile"

    # cat "$mf" | xargs -n 1 -P "$PARALLEL" wget -c -nv -P "$outdir" >> "$logfile" 2>&1
    set +e
    xargs -a "$mf" -n 1 -P "$PARALLEL" wget -c -nv -P "$outdir" >> "$logfile" 2>&1
    dl_status=$?
    set -e
    
    expected=$(wc -l < "$mf")
    got=$(find "$outdir" -maxdepth 1 -type f -name '*.export.CSV.zip' | wc -l)

    echo "expected=$expected got=$got" | tee -a "$logfile"

    if [[ "$expected" -eq "$got" ]]; then
        touch "$doneflag"
        echo "[$(date)] done $month" | tee -a "$logfile"
    else
        echo "[$(date)] WARNING: file count mismatch for $month" | tee -a "$logfile"
    fi
done

echo "[$(date)] all available months processed."

# sbatch scripts/download_gdelt_10y.sh