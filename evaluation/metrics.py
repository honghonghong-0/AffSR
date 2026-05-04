from __future__ import annotations


def recall_at_k(pred_items: list[int], true_item: int, k: int = 10) -> float:
    topk = pred_items[:k]
    return 1.0 if true_item in topk else 0.0

