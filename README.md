# AffSR: Affective Sequential Recommendation with Emotional Drift Modeling

Sequential recommendation model that captures **affective drift** — the temporal shift in a user's emotional state across review sequences — to improve item matching.

## Overview

AffSR extends SASRec with three affective components:

| Component | Role |
|-----------|------|
| **AffDrift** | Separates long-term (EMA) and short-term (last review) sentiment; computes β_final = (1−α)·β(va_long) + α·β(a_n) |
| **EmotionMoE** | K independent FFN experts gated by β_final; produces affective item representation e_final |
| **CrossAttention** | Aligns user representation r_u with e_final before scoring |

## Requirements

```bash
conda create -n affdrift python=3.10
conda activate affdrift
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install transformers pandas numpy scikit-learn pyyaml tqdm
```

## Data Preparation

### CDs & Vinyl (Amazon)

```bash
# Step 1: K-core filtering + metadata join
python preprocessing/preprocess_cds_vinyl.py

# Step 2: GoEmotions inference → VA scores per review
python preprocessing/emotion_extraction.py \
    --processed_path data/processed/CDs_and_Vinyl_processed.csv

# Step 3: Build sequences + train/valid/test splits
python preprocessing/build_sequences_cds_vinyl.py

# Step 4: Compute IDM scores
python preprocessing/compute_idm.py --data_dir data/processed/cds
python preprocessing/convert_idm.py --data_dir data/processed/cds
```

### Movies & TV (Amazon, 2021–2023)

```bash
# All-in-one: k-core + metadata + GoEmotions + sequences
python preprocessing/preprocess_movies2023.py --device cuda

# IDM scores
python preprocessing/compute_idm.py --data_dir data/processed/movies_tv_2021_2023
python preprocessing/convert_idm.py --data_dir data/processed/movies_tv_2021_2023
```

Expected output per dataset:
```
data/processed/{dataset}/
├── user_map.json
├── item_map.json
├── item_cats.json
├── item_va.json          # {item_idx: {va: [v, a], dist28: [...]}}
├── idm.pkl
├── sequences.pkl
└── splits/
    ├── train.pkl
    ├── valid.pkl
    └── test.pkl
```

## Training

```bash
# Full model — CDs & Vinyl
CUDA_VISIBLE_DEVICES=0 python experiments/run_main.py \
    --data_dir data/processed/cds \
    --dataset cds \
    --lr 1e-3 --batch_size 1024 --dropout 0.5 --epochs 200 --patience 10 --K 4

# Full model — Movies & TV
CUDA_VISIBLE_DEVICES=0 python experiments/run_main.py \
    --data_dir data/processed/movies_tv_2021_2023 \
    --dataset movies \
    --lr 1e-3 --batch_size 1024 --dropout 0.5 --epochs 200 --patience 10 --K 4
```

To reproduce all 18 paper experiments (full model + ablations, both datasets) with automatic GPU scheduling:

```bash
bash experiments/run_paper_experiments.sh
```

## Baselines

SASRec, GRU4Rec, and BERT4Rec baselines via RecBole:

```bash
python experiments/run_recbole_baselines.py --model SASRec  --dataset cds    --gpu 0
python experiments/run_recbole_baselines.py --model GRU4Rec --dataset movies --gpu 1
python experiments/run_recbole_baselines.py --model BERT4Rec --dataset cds   --gpu 2
```

## Ablations

| Flag | Description |
|------|-------------|
| `--no_moe` | Remove EmotionMoE; e_final = e_id |
| `--no_long` | Remove long-term sentiment; β = β(a_n) only |
| `--no_short` | Remove short-term sentiment; β = β(va_long) only |
| `--no_ad` | Replace α with a learnable scalar |
| `--K {1,2,3,5}` | Number of MoE experts |

## Project Structure

```
configs/          YAML config templates
datasets/         Dataset loader (AffSRDataset)
models/
  backbone/       SASRec implementation
  modules/        AffSR, AffDrift, EmotionMoE, CrossAttention
trainers/         Training loop, losses
experiments/      Entry points and experiment scripts
preprocessing/    Data preparation pipeline
evaluation/       Metric utilities
data_analysis/    Motivation and EDA scripts
```
