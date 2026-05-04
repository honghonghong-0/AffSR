#!/bin/bash
# RecBole baseline experiments — GRU4Rec, BERT4Rec, SASRec
# Runs on 2 GPUs in parallel: GPU3=movies, GPU0=cds (override with GPUS env var)
#
# Usage:
#   GPUS="3 0" bash experiments/run_recbole_baselines.sh

source ~/miniconda3/etc/profile.d/conda.sh
conda activate affdrift

PROJ="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJ"
export PYTHONPATH="$PROJ"

IFS=' ' read -ra GPUS_ARR <<< "${GPUS:-3 0}"
GPU_MOVIES=${GPUS_ARR[0]}
GPU_CDS=${GPUS_ARR[1]:-${GPUS_ARR[0]}}

OUTDIR=$PROJ/outputs/baselines
mkdir -p $OUTDIR

run_one() {
    local model=$1
    local dataset=$2
    local gpu=$3
    local name="${model,,}_${dataset}"

    if [ -f "$OUTDIR/${name}/test_results.json" ]; then
        echo "[SKIP]  ${name}"
        return 0
    fi

    echo "[START] GPU${gpu} | ${name}"
    mkdir -p "$OUTDIR/${name}"
    CUDA_VISIBLE_DEVICES=$gpu python -u experiments/run_recbole_baselines.py \
        --model "$model" --dataset "$dataset" --gpu "$gpu" \
        --save_dir outputs/baselines \
        > "$OUTDIR/${name}/train.log" 2>&1
    echo "[DONE]  GPU${gpu} | ${name}"
}

# movies sequential (GPU_MOVIES)
run_movies() {
    for model in GRU4Rec BERT4Rec SASRec; do
        run_one "$model" movies $GPU_MOVIES
    done
}

# cds sequential (GPU_CDS)
run_cds() {
    for model in GRU4Rec BERT4Rec SASRec; do
        run_one "$model" cds $GPU_CDS
    done
}

run_movies &
run_cds &
wait

echo ""
echo "=== RecBole baselines done ==="
printf "%-25s  %8s  %8s  %8s  %8s\n" "Experiment" "R@10" "N@10" "R@20" "N@20"
echo "─────────────────────────────────────────────────────"
for model in gru4rec bert4rec sasrec; do
    for ds in movies cds; do
        name="${model}_${ds}"
        result="$OUTDIR/${name}/test_results.json"
        if [ -f "$result" ]; then
            python3 -c "
import json
d = json.load(open('$result'))
print(f\"{'$name':<25}  {d.get('Recall@10',0):.4f}    {d.get('NDCG@10',0):.4f}    {d.get('Recall@20',0):.4f}    {d.get('NDCG@20',0):.4f}\")
" 2>/dev/null
        fi
    done
done
