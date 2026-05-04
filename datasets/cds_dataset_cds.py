"""
datasets/cds_dataset_cds.py
=======================
Amazon CDs & Vinyl dataset loader

Input files (data/processed/cds/):
  - sequences.pkl   : {user_idx: [(item_idx, timestamp, valence, arousal), ...]}
  - splits/train.pkl: {user_idx: [(item_idx, valence, arousal), ...]}
  - splits/valid.pkl: {user_idx: (item_idx, valence, arousal)}
  - splits/test.pkl : {user_idx: (item_idx, valence, arousal)}
  - item_va.json    : {item_idx: {"va": [v, a], "dist28": [...]}}
  - item_map.json   : {parent_asin: item_idx}
"""

import json
import pickle
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


def load_processed_data(data_dir):
    """Load preprocessed data"""
    from pathlib import Path
    data_dir = Path(data_dir)

    with open(data_dir / "sequences.pkl", "rb") as f:
        sequences = pickle.load(f)
    with open(data_dir / "splits" / "train.pkl", "rb") as f:
        train_data = pickle.load(f)
    with open(data_dir / "splits" / "valid.pkl", "rb") as f:
        valid_data = pickle.load(f)
    with open(data_dir / "splits" / "test.pkl", "rb") as f:
        test_data = pickle.load(f)
    with open(data_dir / "item_va.json", "r") as f:
        item_va = json.load(f)
    with open(data_dir / "item_map.json", "r") as f:
        item_map = json.load(f)

    num_items = len(item_map)

    # item_va: convert str keys to int keys
    item_va_arr = {}
    for k, v in item_va.items():
        item_va_arr[int(k)] = v["va"]   # [valence, arousal]

    return sequences, train_data, valid_data, test_data, item_va_arr, num_items


class CDsTrainDataset(Dataset):
    """
    Training dataset

    Each sample:
      - item_seq   : (max_len,)     item sequence (padding=0)
      - a_n        : (2,)           current sentiment state (last review VA)
      - pos_item   : scalar         ground-truth item index
      - e_aff_pos  : (2,)           ground-truth item VA
      - neg_items  : (num_neg,)     negative item indices
      - e_aff_neg  : (num_neg, 2)   negative item VA
    """

    def __init__(self, train_data, sequences, item_va, num_items,
                 max_len=50, num_neg=1, seed=42):
        super().__init__()
        self.train_data = train_data
        self.sequences  = sequences
        self.item_va    = item_va
        self.num_items  = num_items
        self.max_len    = max_len
        self.num_neg    = num_neg
        self.user_ids   = list(train_data.keys())
        self.rng        = random.Random(seed)

        # Build training samples via sliding window
        self.samples = self._build_samples()

    def _build_samples(self):
        """
        Sliding window: generate all (seq[:t], item[t]) pairs
        from each user's train sequence.
        """
        samples = []
        for uid, train_seq in self.train_data.items():
            # train_seq: [(item_idx, valence, arousal), ...]
            if len(train_seq) < 2:
                continue
            for t in range(1, len(train_seq)):
                past_seq = train_seq[:t]       # past sequence (t items)
                target   = train_seq[t]        # target item (t-th, 0-indexed)
                samples.append((uid, past_seq, target))
        return samples

    def _get_seq_tensor(self, past_seq):
        """Pad/trim past sequence to max_len"""
        items = [s[0] for s in past_seq]
        if len(items) > self.max_len:
            items = items[-self.max_len:]
        # Left padding
        pad_len = self.max_len - len(items)
        items   = [0] * pad_len + items
        return torch.tensor(items, dtype=torch.long)

    def _get_a_n(self, past_seq):
        """Current sentiment state: VA of the last review"""
        v, a = past_seq[-1][1], past_seq[-1][2]
        return torch.tensor([v, a], dtype=torch.float32)

    def _sample_negatives(self, pos_item):
        """Random negative sampling, excluding the positive item"""
        negs = []
        while len(negs) < self.num_neg:
            neg = self.rng.randint(1, self.num_items)
            if neg != pos_item:
                negs.append(neg)
        return negs

    def _get_item_va(self, item_idx):
        va = self.item_va.get(item_idx, [0.0, 0.0])
        return torch.tensor(va, dtype=torch.float32)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        uid, past_seq, target = self.samples[idx]
        pos_item = target[0]

        item_seq  = self._get_seq_tensor(past_seq)
        a_n       = self._get_a_n(past_seq)
        e_aff_pos = self._get_item_va(pos_item)

        neg_items = self._sample_negatives(pos_item)
        e_aff_neg = torch.stack([self._get_item_va(n) for n in neg_items])  # (num_neg, 2)

        return {
            "item_seq":   item_seq,
            "a_n":        a_n,
            "pos_item":   torch.tensor(pos_item, dtype=torch.long),
            "e_aff_pos":  e_aff_pos,
            "neg_items":  torch.tensor(neg_items, dtype=torch.long),
            "e_aff_neg":  e_aff_neg,
        }


class CDsEvalDataset(Dataset):
    """
    Evaluation dataset (shared for valid / test)

    Each sample:
      - item_seq   : (max_len,)  item sequence
      - a_n        : (2,)        current sentiment state
      - pos_item   : scalar      ground-truth item index
      - e_aff_pos  : (2,)        ground-truth item VA
    """

    def __init__(self, eval_data, train_data, sequences,
                 item_va, max_len=50):
        super().__init__()
        self.eval_data  = eval_data
        self.train_data = train_data
        self.sequences  = sequences
        self.item_va    = item_va
        self.max_len    = max_len
        self.user_ids   = list(eval_data.keys())

    def _get_seq_tensor(self, uid):
        """Use the full train sequence as input"""
        train_seq = self.train_data.get(uid, [])
        items = [s[0] for s in train_seq]
        if len(items) > self.max_len:
            items = items[-self.max_len:]
        pad_len = self.max_len - len(items)
        items   = [0] * pad_len + items
        return torch.tensor(items, dtype=torch.long)

    def _get_a_n(self, uid):
        """Current sentiment state: VA of the last item in the train sequence"""
        train_seq = self.train_data.get(uid, [])
        if len(train_seq) == 0:
            return torch.zeros(2, dtype=torch.float32)
        v, a = train_seq[-1][1], train_seq[-1][2]
        return torch.tensor([v, a], dtype=torch.float32)

    def _get_item_va(self, item_idx):
        va = self.item_va.get(item_idx, [0.0, 0.0])
        return torch.tensor(va, dtype=torch.float32)

    def __len__(self):
        return len(self.user_ids)

    def __getitem__(self, idx):
        uid      = self.user_ids[idx]
        target   = self.eval_data[uid]
        pos_item = target[0]

        item_seq  = self._get_seq_tensor(uid)
        a_n       = self._get_a_n(uid)
        e_aff_pos = self._get_item_va(pos_item)

        return {
            "user_id":  uid,
            "item_seq": item_seq,
            "a_n":      a_n,
            "pos_item": torch.tensor(pos_item, dtype=torch.long),
            "e_aff_pos": e_aff_pos,
        }


def get_dataloaders(data_dir, max_len=50, num_neg=1,
                    batch_size=256, num_workers=4, seed=42):
    """
    Return train / valid / test DataLoaders.
    """
    sequences, train_data, valid_data, test_data, item_va, num_items = \
        load_processed_data(data_dir)

    train_ds = CDsTrainDataset(
        train_data, sequences, item_va, num_items,
        max_len=max_len, num_neg=num_neg, seed=seed
    )
    valid_ds = CDsEvalDataset(
        valid_data, train_data, sequences, item_va, max_len=max_len
    )
    test_ds  = CDsEvalDataset(
        test_data, train_data, sequences, item_va, max_len=max_len
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              shuffle=True,  num_workers=num_workers,
                              pin_memory=True)
    valid_loader = DataLoader(valid_ds, batch_size=batch_size,
                              shuffle=False, num_workers=num_workers,
                              pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size,
                              shuffle=False, num_workers=num_workers,
                              pin_memory=True)

    return train_loader, valid_loader, test_loader, num_items


if __name__ == "__main__":
    train_loader, valid_loader, test_loader, num_items = get_dataloaders(
        data_dir="data/processed/cds",
        max_len=50, num_neg=1, batch_size=4
    )
    print(f"num_items: {num_items}")
    print(f"train batches: {len(train_loader)}")
    print(f"valid batches: {len(valid_loader)}")
    print(f"test  batches: {len(test_loader)}")

    batch = next(iter(train_loader))
    for k, v in batch.items():
        print(f"  {k}: {v.shape}")
