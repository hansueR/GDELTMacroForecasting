#!/bin/bash
#SBATCH --job-name=TWO_BRANCH
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --partition=bigTiger
#SBATCH --gres=gpu:1
#SBATCH --output=/project/hrao/GDELT/logs_AI/%x_%j.log

set -eo pipefail

source /project/hrao/miniconda3/bin/activate
conda activate comp7720_knn

ROOT=/project/hrao/GDELT
cd "$ROOT"

mkdir -p "$ROOT/results/two_branch_distribution_v1" "$ROOT/logs_AI"

python scripts_AI/10-train_two_branch_distribution_model.py \
  --sequence_dir "$ROOT/modeling/sequence_dataset_l20" \
  --output_dir "$ROOT/results/two_branch_distribution_v1" \
  --first_test_year 2019 \
  --max_horizon 5 \
  --epochs 50 \
  --batch_size 128 \
  --hidden_dim 64 \
  --lstm_layers 1 \
  --dropout 0.1 \
  --asset_emb_dim 8 \
  --lr 0.001 \
  --weight_decay 0.0001 \
  --risk_loss_weight 0.0 \
  --sigma_floor 1e-6 \
  --seed 42 \
  --device cuda

# sbatch scripts_AI/10-train_two_branch_distribution_model.sh