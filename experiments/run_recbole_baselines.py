"""
experiments/run_recbole_baselines.py
=====================================
RecBole baseline experiments for GRU4Rec, BERT4Rec, SASRec

Usage:
  python experiments/run_recbole_baselines.py --model GRU4Rec --dataset movies --gpu 0
  python experiments/run_recbole_baselines.py --model BERT4Rec --dataset cds   --gpu 1
"""

import argparse
import json
import os
import sys
from pathlib import Path

# NumPy 2.0 backward compatibility patch (ray/RecBole uses removed aliases)
import numpy as np
for _old, _new in [('bool8','bool_'),('float_','float64'),('int_','int64'),
                   ('complex_','complex128'),('unicode_','str_'),('unicode','str_')]:
    if not hasattr(np, _old):
        setattr(np, _old, getattr(np, _new))

IDURL_DIR = Path(__file__).parent.parent / "references/IDURL-main/IDURL-main"
# Use standard RecBole (IDURL version has a bug specific to SASRec_IDURL)
os.chdir(IDURL_DIR)

from recbole.quick_start import run_recbole


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",   required=True, choices=["GRU4Rec", "BERT4Rec", "SASRec"])
    parser.add_argument("--dataset", required=True, choices=["movies", "cds"])
    parser.add_argument("--gpu",     default="0")
    parser.add_argument("--save_dir", default="outputs/recbole_baselines")
    args = parser.parse_args()

    run_name = f"{args.model.lower()}_{args.dataset}"
    save_dir = Path(__file__).parent.parent / args.save_dir / run_name
    save_dir.mkdir(parents=True, exist_ok=True)

    result = run_recbole(
        model=args.model,
        dataset=args.dataset,
        config_file_list=[str(IDURL_DIR / "configs/baselines_common.yaml")],
        config_dict={
            "gpu_id":           int(args.gpu),
            "train_batch_size": 512,
            "learning_rate":    0.002,
        },
    )

    # Extract test results
    test_result = result["test_result"]
    out = {
        "Recall@10": float(test_result.get("recall@10", 0)),
        "NDCG@10":   float(test_result.get("ndcg@10", 0)),
        "Recall@20": float(test_result.get("recall@20", 0)),
        "NDCG@20":   float(test_result.get("ndcg@20", 0)),
    }
    with open(save_dir / "test_results.json", "w") as f:
        json.dump(out, f, indent=2)

    print(f"\n[{run_name}] Saved: {save_dir}/test_results.json")
    print(f"  R@10={out['Recall@10']:.4f}  N@10={out['NDCG@10']:.4f}  R@20={out['Recall@20']:.4f}  N@20={out['NDCG@20']:.4f}")


if __name__ == "__main__":
    main()
