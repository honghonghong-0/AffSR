# AffSR

Sequential recommendation model that leverages affective drift in user review sequences.

## Quick start

```bash
conda activate affdrift
cd /path/to/AffSR
python experiments/run_main.py --config configs/base.yaml
```

## What is included

- Config system (`configs/` + `utils/config.py`)
- Training entrypoints (`experiments/run_main.py`, `experiments/run_main_cds.py`)
- Dataset, model, and trainer implementations
- Preprocessing pipeline (`preprocessing/`)
- Evaluation metrics and experiment scripts

## Next

1. Replace placeholder dataset/model with real AffSR logic.
2. Add preprocessing pipeline in `preprocessing/`.
3. Add full evaluation metrics and experiment scripts.
