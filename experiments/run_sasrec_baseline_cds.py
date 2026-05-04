"""
experiments/run_sasrec_baseline_cds.py
=======================================
RecBole SASRec baseline — CDS & Vinyl dataset

Usage:
  python -m experiments.run_sasrec_baseline_cds --gpu_id 1

Metrics: Recall@10, NDCG@10, Recall@20, NDCG@20
"""

import argparse
import pickle
from pathlib import Path

# NumPy 2.0 compatibility patch for RecBole 1.2.1 (run before module loading)
import numpy as np
_np2_aliases = {
    "float_":   "float64",
    "int_":     "int64",
    "complex_": "complex128",
    "bool_":    "bool_",
    "unicode_": "str_",
    "unicode":  "str_",
}
for _alias, _target in _np2_aliases.items():
    if not hasattr(np, _alias):
        setattr(np, _alias, getattr(np, _target))

# Patch torch.load weights_only — also applies to internal RecBole calls
import torch as _torch
_orig_torch_load = _torch.load
def _patched_torch_load(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _orig_torch_load(*args, **kwargs)
_torch.load = _patched_torch_load


# ─────────────────────────────────────────────────────────────────────────────
# sequences.pkl → RecBole .inter file
# ─────────────────────────────────────────────────────────────────────────────
def build_recbole_inter(sequences_path, out_dir):
    """
    sequences.pkl: {user_idx: [(item_idx, timestamp, valence, arousal), ...]}
    → data/recbole/cds/cds.inter  (RecBole SequentialDataset format)
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    inter_path = out_dir / "cds.inter"

    if inter_path.exists():
        print(f"[Data] {inter_path} already exists, reusing")
        return

    with open(sequences_path, "rb") as f:
        sequences = pickle.load(f)

    total = 0
    with open(inter_path, "w") as f:
        f.write("user_id:token\titem_id:token\ttimestamp:float\n")
        for uid, interactions in sequences.items():
            for item_idx, timestamp, _v, _a in interactions:
                f.write(f"{uid}\t{item_idx}\t{timestamp}\n")
                total += 1

    print(f"[Data] .inter file created: {inter_path} ({total:,} rows, "
          f"{len(sequences):,} users)")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",    default="data/processed/cds")
    parser.add_argument("--recbole_dir", default="data/recbole/cds",
                        help="RecBole .inter file save location (data_path/cds/cds.inter)")
    parser.add_argument("--output_dir",  default="outputs/sasrec_baseline_cds")
    parser.add_argument("--gpu_id",      type=int,   default=1)
    parser.add_argument("--epochs",      type=int,   default=200)
    parser.add_argument("--batch_size",  type=int,   default=256)
    parser.add_argument("--lr",          type=float, default=1e-3)
    parser.add_argument("--patience",    type=int,   default=10)
    parser.add_argument("--max_len",     type=int,   default=50)
    parser.add_argument("--d",           type=int,   default=64)
    parser.add_argument("--num_layers",  type=int,   default=2)
    parser.add_argument("--num_heads",   type=int,   default=2)
    parser.add_argument("--dropout",     type=float, default=0.5)
    parser.add_argument("--seed",        type=int,   default=42)
    parser.add_argument("--skip_train",  action="store_true",
                        help="Skip training, run test only with saved checkpoint")
    parser.add_argument("--ckpt",        default=None,
                        help=".pth path for --skip_train (auto-searched in output_dir if not specified)")
    args = parser.parse_args()

    # ── Create .inter file ──────────────────────────────────────────────────
    sequences_path = Path(args.data_dir) / "sequences.pkl"
    build_recbole_inter(sequences_path, args.recbole_dir)

    # ── RecBole import ──────────────────────────────────────────────────────
    from recbole.config import Config
    from recbole.data import create_dataset, data_preparation
    from recbole.model.sequential_recommender import SASRec
    from recbole.utils import init_seed, init_logger, get_trainer

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # recbole_dir = data/recbole/cds  →  data_path = data/recbole
    data_path = str(Path(args.recbole_dir).parent)

    config_dict = {
        # Data path
        "data_path":           data_path,
        "USER_ID_FIELD":       "user_id",
        "ITEM_ID_FIELD":       "item_id",
        "TIME_FIELD":          "timestamp",
        "load_col":            {"inter": ["user_id", "item_id", "timestamp"]},
        # Leave-one-out chronological split (SASRec standard protocol)
        "eval_args": {
            "split":    {"LS": "valid_and_test"},
            "order":    "TO",
            "group_by": "user",
            "mode":     {"valid": "full", "test": "full"},
        },
        # Evaluation metrics
        "metrics":             ["Recall", "NDCG"],
        "topk":                [10, 20],
        "valid_metric":        "NDCG@10",
        # SASRec hyperparameters
        "hidden_size":         args.d,
        "num_attention_heads": args.num_heads,
        "num_hidden_layers":   args.num_layers,
        "hidden_dropout_prob": args.dropout,
        "attn_dropout_prob":   args.dropout,
        "hidden_act":          "gelu",
        "max_seq_length":      args.max_len,
        "loss_type":           "CE",          # full softmax cross-entropy
        "train_neg_sample_args": None,        # CE loss does not require negative sampling
        # Training
        "epochs":              args.epochs,
        "train_batch_size":    args.batch_size,
        "eval_batch_size":     args.batch_size,
        "learning_rate":       args.lr,
        "stopping_step":       args.patience,
        # Environment
        "gpu_id":              str(args.gpu_id),
        "use_gpu":             True,
        "seed":                args.seed,
        "reproducibility":     True,
        "log_wandb":           False,
        "checkpoint_dir":      args.output_dir,
    }

    config = Config(model="SASRec", dataset="cds", config_dict=config_dict)
    init_seed(config["seed"], config["reproducibility"])
    init_logger(config)

    # ── Dataset ─────────────────────────────────────────────────────────────
    print("\n[Data] Preparing RecBole dataset...")
    dataset = create_dataset(config)
    print(dataset)

    train_data, valid_data, test_data = data_preparation(config, dataset)

    # ── Model ────────────────────────────────────────────────────────────────
    print("\n[Model] Initializing SASRec...")
    model = SASRec(config, train_data.dataset).to(config["device"])
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameter count: {n_params:,}")
    print(f"  hidden_size={args.d}, layers={args.num_layers}, "
          f"heads={args.num_heads}, max_len={args.max_len}, dropout={args.dropout}")

    # ── Training ─────────────────────────────────────────────────────────────
    trainer = get_trainer(config["MODEL_TYPE"], config["model"])(config, model)

    if args.skip_train:
        # Search for checkpoint
        ckpt_path = args.ckpt
        if ckpt_path is None:
            pth_files = sorted(Path(args.output_dir).glob("*.pth"))
            if not pth_files:
                raise FileNotFoundError(
                    f"No checkpoint found: {args.output_dir}/*.pth\n"
                    "Specify a path with --ckpt or run without --skip_train."
                )
            ckpt_path = str(pth_files[-1])
        print(f"\n[Skip Train] Loading checkpoint: {ckpt_path}")
    else:
        print("\n[Train] Starting training...")
        best_valid_score, best_valid_result = trainer.fit(
            train_data, valid_data, saved=True, verbose=True
        )
        ckpt_path = None  # auto-loaded with load_best_model=True in evaluate

    # ── Test ─────────────────────────────────────────────────────────────────
    print("\n[Test] Evaluating...")
    # load_best_model=True: uses self.saved_model_file when model_file is None
    test_result = trainer.evaluate(
        test_data,
        load_best_model=True,
        model_file=ckpt_path,
        show_progress=True,
    )

    # ── Print results ─────────────────────────────────────────────────────────
    print("\n" + "=" * 50)
    print("SASRec Baseline Results  (CDS & Vinyl)")
    print("=" * 50)
    target_metrics = ["recall@10", "ndcg@10", "recall@20", "ndcg@20"]
    for key in target_metrics:
        val = test_result.get(key, test_result.get(key.upper(), "N/A"))
        label = key.upper().replace("@", "@")
        if isinstance(val, float):
            print(f"  {label:12s}: {val:.4f}")
        else:
            print(f"  {label:12s}: {val}")
    print("=" * 50)
    print(f"\n[Save] Checkpoint: {args.output_dir}")


if __name__ == "__main__":
    main()
