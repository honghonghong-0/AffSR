#!/bin/bash
# run_v10_distfix.sh
# Re-run all experiments after fixing the dist28 bug (cds_v10 / movies_v10)

set -e
source ~/miniconda3/etc/profile.d/conda.sh
conda activate affdrift

PROJ="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJ"
export PYTHONPATH="$PROJ:$PYTHONPATH"

SAVE_DIR="outputs/v10_distfix"
mkdir -p "$SAVE_DIR"

COMMON="--batch_size 512 --lr 2e-3 --weight_decay 0.001 --full_ce
        --epochs 200 --patience 10 --max_seq_len 50
        --d_model 64 --n_heads 2 --n_layers 2 --dropout 0.2 --K 4"

MOVIES="--data_dir data/processed/movies_v10 --dataset movies"
CDS="--data_dir data/processed/cds_v10    --dataset cds"

run_exp() {
  local RUN_NAME=$1; shift
  mkdir -p "$SAVE_DIR/$RUN_NAME"
  python -u experiments/run_main.py \
    $COMMON "$@" \
    --run_name "$RUN_NAME" --save_dir "$SAVE_DIR" \
    2>&1 | tee "$SAVE_DIR/$RUN_NAME/train.log"
}

# ── GPU 1: affsr_full_movies / no_long / no_short ──────────────────────────
(
  CUDA_VISIBLE_DEVICES=1 run_exp affsr_full_movies        $MOVIES
  CUDA_VISIBLE_DEVICES=1 run_exp ablation_no_long_movies  $MOVIES --no_long
  CUDA_VISIBLE_DEVICES=1 run_exp ablation_no_short_movies $MOVIES --no_short
  echo "[GPU 1] Done"
) &
GPU1_PID=$!
echo "[GPU 1] PID=$GPU1_PID | affsr_full_movies -> no_long -> no_short"

# ── GPU 2: affsr_full_cds / no_ad_movies / no_moe_movies ───────────────────
(
  CUDA_VISIBLE_DEVICES=2 run_exp affsr_full_cds          $CDS
  CUDA_VISIBLE_DEVICES=2 run_exp ablation_no_ad_movies   $MOVIES --no_ad
  CUDA_VISIBLE_DEVICES=2 run_exp ablation_no_moe_movies  $MOVIES --no_moe
  echo "[GPU 2] Done"
) &
GPU2_PID=$!
echo "[GPU 2] PID=$GPU2_PID | affsr_full_cds -> no_ad_movies -> no_moe_movies"

# ── GPU 3: ablation 4 runs (cds) ───────────────────────────────────────────
(
  CUDA_VISIBLE_DEVICES=3 run_exp ablation_no_long_cds   $CDS --no_long
  CUDA_VISIBLE_DEVICES=3 run_exp ablation_no_short_cds  $CDS --no_short
  CUDA_VISIBLE_DEVICES=3 run_exp ablation_no_ad_cds     $CDS --no_ad
  CUDA_VISIBLE_DEVICES=3 run_exp ablation_no_moe_cds    $CDS --no_moe
  echo "[GPU 3] Done"
) &
GPU3_PID=$!
echo "[GPU 3] PID=$GPU3_PID | no_long_cds -> no_short_cds -> no_ad_cds -> no_moe_cds"

echo ""
echo "Starting 10 experiments. Logs: $SAVE_DIR/{run_name}/train.log"
wait $GPU1_PID $GPU2_PID $GPU3_PID
echo "All experiments completed."
