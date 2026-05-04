#!/bin/bash
# Run all paper experiments (18 runs)
# Auto-detect free GPUs and assign dynamically (max 1 job per GPU)

source ~/miniconda3/etc/profile.d/conda.sh
conda activate affdrift

PROJ="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJ"
export PYTHONPATH="$PROJ"

OUT=$PROJ/outputs
COMMON="--lr 1e-3 --batch_size 1024 --dropout 0.5 --epochs 200 --patience 10"
MIN_FREE_MEM=5000  # MiB — only use GPUs with at least this much free memory

# ── GPU pool (auto-detect free GPUs) ─────────────────────────────────────────
get_free_gpus() {
    nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits \
        | awk -F', ' -v min=$MIN_FREE_MEM '$2 >= min {print $1}'
}

FREE_GPUS=($(get_free_gpus))
if [ ${#FREE_GPUS[@]} -eq 0 ]; then
    echo "[ERROR] No GPU with >=${MIN_FREE_MEM}MiB free memory found."
    exit 1
fi
echo "=== Available GPUs: ${FREE_GPUS[*]} ==="

# Named pipe GPU semaphore (max 1 concurrent job per GPU)
FIFO=$(mktemp -u)
mkfifo "$FIFO"
exec 3<>"$FIFO"
rm -f "$FIFO"
for gpu in "${FREE_GPUS[@]}"; do echo $gpu; done >&3

# ── Experiment runner ────────────────────────────────────────────────────────
run_experiment() {
    local name=$1
    local dataset=$2
    shift 2

    local data_dir
    if   [ "$dataset" = "cds"    ]; then data_dir="data/processed/cds"
    elif [ "$dataset" = "movies" ]; then data_dir="data/processed/movies_tv_2021_2023"
    else echo "[ERROR] Unknown dataset: $dataset"; return 1; fi

    # Acquire GPU
    local gpu
    read -u 3 gpu
    if [ -z "$gpu" ]; then
        echo "[ERROR] GPU allocation failed: ${name} — skipping"
        return 1
    fi

    mkdir -p "$OUT/${name}"
    echo "[START] GPU${gpu} | ${name}"
    CUDA_VISIBLE_DEVICES=$gpu python -u experiments/run_main.py \
        --data_dir "$data_dir" --dataset "$dataset" \
        $COMMON "$@" \
        --run_name "${name}" \
        --save_dir "$OUT/${name}" \
        2>&1 | sed 's/\x1b\[[0-9;]*m//g' | tee "$OUT/${name}/train.log" > /dev/null
    echo "[DONE]  GPU${gpu} | ${name}"

    # Release GPU
    echo $gpu >&3
}

# ── Define and submit 18 experiments ────────────────────────────────────────
# CDS (9 runs)
run_experiment affsr_full_cds          cds    --K 4              &
run_experiment ablation_no_long_cds    cds    --K 4 --no_long    &
run_experiment ablation_no_short_cds   cds    --K 4 --no_short   &
run_experiment ablation_no_ad_cds      cds    --K 4 --no_ad      &
run_experiment ablation_no_moe_cds     cds    --K 4 --no_moe     &
run_experiment ablation_K1_cds         cds    --K 1              &
run_experiment ablation_K2_cds         cds    --K 2              &
run_experiment ablation_K3_cds         cds    --K 3              &
run_experiment ablation_K5_cds         cds    --K 5              &

# Movies (9 runs)
run_experiment affsr_full_movies       movies --K 4              &
run_experiment ablation_no_long_movies movies --K 4 --no_long    &
run_experiment ablation_no_short_movies movies --K 4 --no_short  &
run_experiment ablation_no_ad_movies   movies --K 4 --no_ad      &
run_experiment ablation_no_moe_movies  movies --K 4 --no_moe     &
run_experiment ablation_K1_movies      movies --K 1              &
run_experiment ablation_K2_movies      movies --K 2              &
run_experiment ablation_K3_movies      movies --K 3              &
run_experiment ablation_K5_movies      movies --K 5              &

wait
echo "=== All experiments done ==="
