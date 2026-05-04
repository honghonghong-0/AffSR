"""
datasets/base_dataset.py
========================
Dataset class for AffSR model (v10)

Input files:
  - splits/train.pkl  : {user_idx: [(item_idx, v, a), ...]}
  - splits/valid.pkl  : {user_idx: (item_idx, v, a)}
  - splits/test.pkl   : {user_idx: (item_idx, v, a)}
  - item_va.json      : {item_idx: {"va": [v, a], "dist28": [...]}}
  - item_cats.json    : {item_idx: [cat1, cat2, ...]}
  - idm.pkl           : {(user_idx, target_item_idx): idm_score}

Batch output:
  item_seq     : (B, L)       integer item ID sequence (with padding)
  va_seq       : (B, L, 2)    user VA sequence (actual review sentiment from sequences.pkl)
  dist28_seq   : (B, L, 28)   GoEmotions 28-dim per review (for EMA long-term sentiment)
  seq_mask     : (B, L)       padding mask (True = valid)
  a_n          : (B, 2)       current sentiment (last valid timestep, from sequences.pkl)
  e_aff_pos    : (B, 2)       positive item VA
  idm          : (B,)         IDM scalar
  target       : (B,)         ground-truth item ID
  neg_items    : (B, num_neg) negative item IDs (for training)
"""

import json
import pickle
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


class AffSRDataset(Dataset):
    """
    Dataset for AffSR training and evaluation.

    split='train' : generates (seq, target) pairs via sliding window
    split='valid' : the second-to-last item per user is the target
    split='test'  : the last item per user is the target
    """

    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        max_seq_len: int = 50,
        num_neg: int = 1,
        seed: int = 42,
        full_ce: bool = False,
    ):
        assert split in ("train", "valid", "test")
        self.split = split
        self.max_seq_len = max_seq_len
        self.num_neg = num_neg
        self.full_ce = full_ce
        self.rng = random.Random(seed)

        data_dir = Path(data_dir)

        # ── Load split data ───────────────────────────────────────────
        with open(data_dir / "splits" / f"{split}.pkl", "rb") as f:
            self.split_data = pickle.load(f)

        # train.pkl : {user_idx: [(item_idx, v, a), ...]}  ← full train sequence
        # valid/test.pkl : {user_idx: (item_idx, v, a)}    ← single target
        #
        # For train split: sliding window samples are constructed directly.
        # For valid/test: train portion of sequences.pkl is used as input.
        with open(data_dir / "splits" / "train.pkl", "rb") as f:
            self.train_seqs: Dict[int, List] = pickle.load(f)

        # RecBole protocol: include valid item in sequence during test
        if split == "test":
            with open(data_dir / "splits" / "valid.pkl", "rb") as f:
                self.valid_targets: Dict[int, tuple] = pickle.load(f)
        else:
            self.valid_targets = {}

        # ── Load item VA / dist28 ────────────────────────────────────
        with open(data_dir / "item_va.json") as f:
            raw_va = json.load(f)
        self.item_va: Dict[int, np.ndarray] = {
            int(k): np.array(v["va"], dtype=np.float32)
            for k, v in raw_va.items()
            if k != "__meta__"
        }

        # ── Load IDM ─────────────────────────────────────────────────
        idm_path = data_dir / "idm.pkl"
        if idm_path.exists():
            with open(idm_path, "rb") as f:
                self.idm: Dict[Tuple[int, int], float] = pickle.load(f)
        else:
            # Fill with 0 if IDM file is missing (temporary fallback when preprocessing is incomplete)
            print("[AffSRDataset] Warning: idm.pkl not found → filling IDM=0.0")
            self.idm = {}

        # ── Full item set (for negative sampling) ────────────────────
        self.all_items: List[int] = list(self.item_va.keys())
        self.num_items = max(self.all_items) + 1  # 0 is padding

        # ── Build training samples (train split only) ────────────────
        if split == "train":
            self.samples = self._build_train_samples()
        else:
            self.samples = self._build_eval_samples()

    # ─────────────────────────────────────────────────────────────────
    # Sample construction
    # ─────────────────────────────────────────────────────────────────

    def _build_train_samples(self) -> List[dict]:
        """
        Generate (input sequence, target) pairs via sliding window.

        Creates samples from train.pkl sequences using each timestep
        t=1,...,|seq|-1 as the target.
        Minimum sequence length is 2 (1 input + 1 target).
        """
        samples = []
        for user_idx, seq in self.train_seqs.items():
            if len(seq) < 2:
                continue
            # seq: [(item_idx, v, a, ...), ...]
            for t in range(1, len(seq)):
                input_seq = seq[max(0, t - self.max_seq_len): t]
                target = seq[t]
                samples.append({
                    "user_idx":   user_idx,
                    "input_seq":  input_seq,
                    "target_idx": target[0],
                    "target_va":  (target[1], target[2]),
                })
        return samples

    def _build_eval_samples(self) -> List[dict]:
        """
        valid/test: Use the full train sequence per user as input,
        with the target item from split_data as the ground truth.
        """
        samples = []
        for user_idx, target_tuple in self.split_data.items():
            if user_idx not in self.train_seqs:
                continue
            seq = self.train_seqs[user_idx]
            if len(seq) == 0:
                continue
            # target_tuple: (item_idx, v, a)
            target_idx, target_v, target_a = target_tuple[0], target_tuple[1], target_tuple[2]
            # RecBole protocol: append valid item to end of sequence during test
            if self.split == "test" and user_idx in self.valid_targets:
                full_seq = list(seq) + [self.valid_targets[user_idx]]
            else:
                full_seq = list(seq)
            input_seq = full_seq[-self.max_seq_len:]
            samples.append({
                "user_idx":   user_idx,
                "input_seq":  input_seq,
                "target_idx": target_idx,
                "target_va":  (target_v, target_a),
            })
        return samples

    # ─────────────────────────────────────────────────────────────────
    # __len__ / __getitem__
    # ─────────────────────────────────────────────────────────────────


    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]
        user_idx   = sample["user_idx"]
        input_seq  = sample["input_seq"]   # [(item_idx, v, a), ...]
        target_idx = sample["target_idx"]

        L = len(input_seq)

        # ── Sequence padding (left-padding, SASRec convention) ────────
        item_seq   = np.zeros(self.max_seq_len, dtype=np.int64)
        va_seq     = np.zeros((self.max_seq_len, 2), dtype=np.float32)
        dist28_seq = np.zeros((self.max_seq_len, 28), dtype=np.float32)
        mask       = np.zeros(self.max_seq_len, dtype=bool)

        pad_len = self.max_seq_len - L
        for i, item_tuple in enumerate(input_seq):
            # cds_v10:    (item_idx, v, a, dist28)           → dist28 at index 3
            # movies_v10: (item_idx, v, a, has_va, dist28)   → dist28 at index 4
            item_i, v_i, a_i = item_tuple[0], item_tuple[1], item_tuple[2]
            item_seq[pad_len + i] = item_i
            va_seq[pad_len + i]   = [v_i, a_i]
            if len(item_tuple) >= 5:
                dist28_seq[pad_len + i] = np.asarray(item_tuple[4], dtype=np.float32)
            elif len(item_tuple) >= 4:
                dist28_seq[pad_len + i] = np.asarray(item_tuple[3], dtype=np.float32)
            mask[pad_len + i] = True

        # ── a_n: sentiment at the last valid timestep (actual review VA from sequences.pkl) ─
        a_n = va_seq[pad_len + L - 1].copy()  # (2,)

        # ── e_aff (positive target item VA) ──────────────────────────
        e_aff_pos = self.item_va.get(
            target_idx, np.zeros(2, dtype=np.float32)
        )

        # ── IDM ─────────────────────────────────────────────────────
        idm_score = float(
            self.idm.get((user_idx, target_idx), 0.0)
        )

        batch = {
            "item_seq":    torch.from_numpy(item_seq),         # (L,)
            "va_seq":      torch.from_numpy(va_seq),           # (L, 2)
            "dist28_seq":  torch.from_numpy(dist28_seq),       # (L, 28)
            "seq_mask":    torch.from_numpy(mask),             # (L,)
            "a_n":         torch.from_numpy(a_n),              # (2,)
            "e_aff_pos":   torch.from_numpy(e_aff_pos),       # (2,)
            "idm":         torch.tensor(idm_score, dtype=torch.float32),
            "target":      torch.tensor(target_idx, dtype=torch.long),
            "user_idx":    torch.tensor(user_idx, dtype=torch.long),
        }

        if not self.full_ce:
            seen = {s[0] for s in input_seq} | {target_idx}
            neg_items = self._sample_negatives(seen, self.num_neg)
            batch["neg_items"] = torch.tensor(neg_items, dtype=torch.long)

        return batch

    # ─────────────────────────────────────────────────────────────────
    # Negative sampling
    # ─────────────────────────────────────────────────────────────────

    def _sample_negatives(self, seen: set, n: int) -> List[int]:
        """Uniform negative sampling, excluding seen items."""
        negs = []
        while len(negs) < n:
            cand = self.rng.choice(self.all_items)
            if cand not in seen:
                negs.append(cand)
        return negs

    # ─────────────────────────────────────────────────────────────────
    # Utilities
    # ─────────────────────────────────────────────────────────────────

    def get_item_va(self, item_idx: int) -> np.ndarray:
        """Retrieve item VA (used externally outside the model)."""
        return self.item_va.get(item_idx, np.zeros(2, dtype=np.float32))

    def __repr__(self) -> str:
        return (
            f"AffSRDataset(split={self.split}, "
            f"samples={len(self.samples)}, "
            f"num_items={self.num_items}, "
            f"max_seq_len={self.max_seq_len})"
        )