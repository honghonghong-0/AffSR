"""
plot_va_cds.py
==============
emotion_results_cds.csv 캐시 기반으로 VA 시각화만 뽑는 스크립트
- VA scatter (사분면별 색상)
- VA Quadrant Distribution 바차트

사용법:
  python preprocessing/plot_va_cds.py \
      --results_path data_analysis/results/emotion_results_cds.csv \
      --output_dir   data_analysis/results
"""

import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

QUADRANT_INFO = {
    "Q1 (High V, High A)": {"color": "#E8A838", "short": "Q1\nHigh V\nHigh A"},
    "Q2 (Low V, High A)":  {"color": "#E85252", "short": "Q2\nLow V\nHigh A"},
    "Q3 (High V, Low A)":  {"color": "#52B788", "short": "Q3\nHigh V\nLow A"},
    "Q4 (Low V, Low A)":   {"color": "#6B8CBA", "short": "Q4\nLow V\nLow A"},
}

def get_quadrant(v, a):
    if   v >= 0 and a >= 0: return "Q1 (High V, High A)"
    elif v <  0 and a >= 0: return "Q2 (Low V, High A)"
    elif v >= 0 and a <  0: return "Q3 (High V, Low A)"
    else:                   return "Q4 (Low V, Low A)"


def plot_va_scatter(df, out_path):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.axhline(0, color="gray", lw=0.8, ls="--")
    ax.axvline(0, color="gray", lw=0.8, ls="--")

    for q, info in QUADRANT_INFO.items():
        mask = df["quadrant"] == q
        ax.scatter(
            df[mask]["valence"], df[mask]["arousal"],
            alpha=0.3, s=10, color=info["color"],
            label=f"{q} ({mask.sum()})"
        )
    ax.set_xlabel("Valence", fontsize=10)
    ax.set_ylabel("Arousal", fontsize=10)
    ax.set_title("VA Distribution — CDs & Vinyl", fontsize=11)
    ax.legend(fontsize=8, loc="lower right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Plot] VA scatter → {out_path}")


def plot_va_quadrant_bar(df, out_path):
    counts = df["quadrant"].value_counts()
    total  = counts.sum()

    # 사분면 순서 고정
    order  = list(QUADRANT_INFO.keys())
    values = [counts.get(q, 0) / total * 100 for q in order]
    colors = [QUADRANT_INFO[q]["color"] for q in order]
    labels = [QUADRANT_INFO[q]["short"] for q in order]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(labels, values, color=colors, edgecolor="white", width=0.5)

    # 수치 레이블
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.5,
                f"{val:.1f}%", ha="center", va="bottom", fontsize=10)

    # Uniform 기준선
    ax.axhline(25, color="gray", lw=1.2, ls="--", label="Uniform (25%)")
    ax.set_ylabel("Proportion (%)", fontsize=10)
    ax.set_title("VA Quadrant Distribution (neutral excluded)", fontsize=11)
    ax.legend(fontsize=9)
    ax.set_ylim(0, max(values) * 1.15)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Plot] VA quadrant bar → {out_path}")

    # 분포 출력
    print("\n[VA Quadrant 분포]")
    for q, v in zip(order, values):
        print(f"  {q}: {v:.1f}%")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_path", default="data_analysis/results/emotion_results_cds.csv")
    parser.add_argument("--output_dir",   default="data_analysis/results")
    args = parser.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    p = lambda name: str(Path(args.output_dir) / name)

    print(f"[Load] {args.results_path}")
    df = pd.read_csv(args.results_path)

    # quadrant 컬럼 없으면 계산
    if "quadrant" not in df.columns:
        df["quadrant"] = [get_quadrant(v, a)
                          for v, a in zip(df["valence"], df["arousal"])]

    plot_va_scatter(df, p("va_scatter_cds.png"))
    plot_va_quadrant_bar(df, p("va_quadrant_bar_cds.png"))
    print("\nDone ✓")


if __name__ == "__main__":
    main()
