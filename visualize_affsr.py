"""
visualize_affsr.py
==================
Figure generation for AffSR paper

Panel (a): User emotion trajectory in VA space (short-term a_n vs long-term va_long)
Panel (b): MoE β weights by emotional state (Russell quadrants)
Panel (c): Shift in recommended item VA distribution by emotional state

Usage:
  python visualize_affsr.py \
    --ckpt outputs/v10_final/affsr_full_movies/best.pt \
    --data_dir data/processed/movies_tv_2021_2023 \
    --dataset movies \
    --out_dir outputs/figures
"""

import argparse
import json
import pickle
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import torch
from torch.utils.data import DataLoader
from scipy.ndimage import gaussian_filter

# ── Global style ──────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":       "Arial",
    "axes.facecolor":    "white",
    "figure.facecolor":  "white",
    "axes.edgecolor":    "#AAAAAA",
    "axes.linewidth":    0.8,
    "xtick.labelsize":   9,
    "ytick.labelsize":   9,
})

# Custom pastel cmap (blue→white→red)
PASTEL_CMAP = LinearSegmentedColormap.from_list(
    "pastel_rdbu",
    ["#4393C3", "#92C5DE", "#D1E5F0", "#FFFFFF", "#FDDBC7", "#F4A582", "#D6604D"],
    N=256,
)

from datasets.base_dataset import AffSRDataset
from models.modules.affsr import AffSR


# ── GoEmotions VA matrix (same as AffDrift) ───────────────────────────────────
NEUTRAL_IDX = 27
GOEMO_VA = [
    ( 0.69,  0.55), ( 0.50,  0.20), ( 0.10, -0.30), ( 0.70,  0.60),
    (-0.40, -0.10), ( 0.40,  0.10), ( 0.40, -0.30), (-0.50,  0.30),
    (-0.30,  0.10), ( 0.30, -0.10), ( 0.80,  0.70), ( 0.20,  0.30),
    ( 0.50,  0.30), (-0.60, -0.30), (-0.60, -0.40), (-0.40,  0.20),
    ( 0.60,  0.40), ( 0.60,  0.50), (-0.70, -0.50), ( 0.30,  0.10),
    ( 0.30, -0.20), (-0.50, -0.20), ( 0.20,  0.50), ( 0.70,  0.30),
    ( 0.40,  0.10), (-0.30, -0.10), ( 0.50,  0.20), ( 0.00,  0.00),
]
QUADRANT_NAMES   = ["Q1\n(+V+A)", "Q2\n(-V+A)", "Q3\n(-V-A)", "Q4\n(+V-A)"]
QUADRANT_COLORS  = ["#D6604D", "#74ADD1", "#4575B4", "#92C5DE"]
QUADRANT_EMOTIONS = ["Joy\nExcitement", "Anger\nFear", "Sadness\nDisgust", "Calm\nRelief"]


def load_model(ckpt_path: str, data_dir: str, device: torch.device,
               no_moe: bool = False) -> tuple:
    train_ds = AffSRDataset(data_dir, split="train", max_seq_len=50, full_ce=True)
    num_items = train_ds.num_items

    model = AffSR(num_items=num_items, d_model=64, n_heads=2, n_layers=2,
                  max_seq_len=50, K=4, dropout=0.0, no_moe=no_moe)
    state = torch.load(ckpt_path, map_location=device)
    model_state = state["model"] if "model" in state else state
    model.load_state_dict(model_state)
    model.to(device)
    model.eval()
    return model, train_ds


def extract_user_states(model: AffSR, data_dir: str, device: torch.device,
                        n_users: int = 2000) -> dict:
    """Extract emotional states for users in the test set."""
    test_ds = AffSRDataset(data_dir, split="test", max_seq_len=50, full_ce=True)
    loader = DataLoader(test_ds, batch_size=128, shuffle=False, num_workers=0)

    # Load item VA
    with open(Path(data_dir) / "item_va.json") as f:
        raw_va = json.load(f)
    item_va = {int(k): v["va"] for k, v in raw_va.items() if k != "__meta__"}

    results = {
        "a_n": [], "va_long": [], "beta": [], "adm": [],
        "topk_va": [], "user_idx": [],
    }

    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            B = batch["item_seq"].size(0)

            r_u = model.sasrec(batch["item_seq"], batch["seq_mask"])
            adm, beta = model.affdrift(batch["a_n"], batch["dist28_seq"], batch["seq_mask"])

            # Compute va_long directly (reproducing affdrift internal logic)
            L = batch["dist28_seq"].size(1)
            t = torch.arange(L, device=device, dtype=torch.float32)
            w = torch.exp(-torch.nn.functional.softplus(model.affdrift.lambda_raw) * (L - 1 - t))
            w = w * batch["seq_mask"].float()
            w = w / (w.sum(dim=1, keepdim=True) + 1e-8)
            d28 = batch["dist28_seq"].clone()
            d28[:, :, NEUTRAL_IDX] = 0.0
            d28_sum = d28.sum(dim=-1, keepdim=True).clamp(min=1e-8)
            d28 = d28 / d28_sum
            h_long = (w.unsqueeze(-1) * d28).sum(dim=1)
            va_long = h_long @ model.affdrift.va_matrix

            # Top-10 recommended item VA (full predict — includes MoE emotion conditioning)
            all_item_va = torch.zeros(model.num_items + 1, 2, device=device)
            scores = model.predict(
                item_seq=batch["item_seq"],
                seq_mask=batch["seq_mask"],
                a_n=batch["a_n"],
                dist28_seq=batch["dist28_seq"],
                idm=batch["idm"],
                all_item_va=all_item_va,
                chunk_size=512,
            )
            scores[:, 0] = float("-inf")
            seen = batch["item_seq"]
            for b in range(B):
                scores[b, seen[b]] = float("-inf")
            topk = scores.topk(10, dim=-1).indices.cpu().numpy()

            for b in range(B):
                topk_va_list = [item_va[int(i)] for i in topk[b] if int(i) in item_va]
                if topk_va_list:
                    results["topk_va"].append(np.mean(topk_va_list, axis=0))
                else:
                    results["topk_va"].append(np.zeros(2))

            results["a_n"].extend(batch["a_n"].cpu().numpy())
            results["va_long"].extend(va_long.cpu().numpy())
            results["beta"].extend(beta.cpu().numpy())
            results["adm"].extend(adm.cpu().numpy())
            results["user_idx"].extend(batch["user_idx"].cpu().numpy())

            if len(results["a_n"]) >= n_users:
                break

    for k in ["a_n", "va_long", "beta", "adm", "topk_va"]:
        results[k] = np.array(results[k][:n_users])

    # ── Score Shift: item score difference between Joy(Q1) and Sad(Q3) states ─
    # Take one representative batch, override β with Q1/Q3, and compare scores
    print("Computing score shift (Joy vs Sad)...")
    sample_batch = next(iter(DataLoader(
        AffSRDataset(data_dir, split="test", max_seq_len=50, full_ce=True),
        batch_size=32, shuffle=True, num_workers=0,
    )))
    sample_batch = {k: v.to(device) for k, v in sample_batch.items()}
    B32 = sample_batch["item_seq"].size(0)
    N = model.num_items + 1
    d = model.d_model

    all_item_va_zero = torch.zeros(N, 2, device=device)

    # Build item VA array first (needed for penalty computation)
    item_va_arr = np.zeros((N, 2), dtype=np.float32)
    for idx, va in item_va.items():
        if idx < N:
            item_va_arr[idx] = va

    # Extract Joy/Sad representative a_n values from learned centroids
    centroids_np = model.affdrift.centroids.detach().cpu().numpy()  # (K, 2)
    # Centroid with highest +V+A is Joy, lowest is Sad
    joy_idx = int(np.argmax(centroids_np[:, 0] + centroids_np[:, 1]))
    sad_idx = int(np.argmin(centroids_np[:, 0] + centroids_np[:, 1]))
    a_n_joy_np = centroids_np[joy_idx]   # e.g. [0.83, 0.61]
    a_n_sad_np = centroids_np[sad_idx]   # e.g. [-0.69, -0.43]
    a_n_joy_t  = torch.tensor(a_n_joy_np, device=device, dtype=torch.float32)
    a_n_sad_t  = torch.tensor(a_n_sad_np, device=device, dtype=torch.float32)
    lambda_mc  = torch.nn.functional.softplus(model.lambda_mc).item()

    with torch.no_grad():
        r_u = model.sasrec(sample_batch["item_seq"], sample_batch["seq_mask"])
        r_bar_u = model._user_repr(r_u, torch.zeros(B32, model.K, device=device))

        def scores_with_beta_and_an(beta_idx, a_n_t):
            """Full score including MoE score + penalty term."""
            beta_fixed = torch.zeros(B32, model.K, device=device)
            beta_fixed[:, beta_idx] = 1.0
            chunks = []
            for start in range(0, N, 512):
                end = min(start + 512, N)
                C = end - start
                item_ids_c = torch.arange(start, end, device=device)
                e_id_c = model.item_emb(item_ids_c)
                e_id_flat   = e_id_c.unsqueeze(0).expand(B32, C, d).reshape(B32*C, d)
                beta_exp    = beta_fixed.unsqueeze(1).expand(B32, C, model.K).reshape(B32*C, model.K)
                r_bar_u_exp = r_bar_u.unsqueeze(1).expand(B32, C, d).reshape(B32*C, d)
                sc = model._chunk_forward(e_id_flat, beta_exp, r_bar_u_exp)
                chunks.append(sc.view(B32, C))
            moe_scores = torch.cat(chunks, dim=1).mean(dim=0)  # (N,)

            # penalty: softplus(λ) * ||va_item - a_n||
            all_va_t = torch.tensor(item_va_arr, device=device, dtype=torch.float32)  # (N, 2)
            penalty = lambda_mc * torch.norm(all_va_t - a_n_t.unsqueeze(0), dim=-1)   # (N,)
            return (moe_scores - penalty).cpu().numpy()

        scores_joy = scores_with_beta_and_an(joy_idx, a_n_joy_t)
        scores_sad = scores_with_beta_and_an(sad_idx, a_n_sad_t)
        score_shift = scores_joy - scores_sad  # (N,): positive = Joy-preferred, negative = Sad-preferred

        # Top-20 item VA per quadrant β (for Ellipse)
        top20_va_per_quadrant = []
        for q in range(4):
            a_n_q = torch.tensor(centroids_np[q], device=device, dtype=torch.float32)
            sc_q = scores_with_beta_and_an(q, a_n_q)
            top20_idx = np.argsort(sc_q)[::-1][:20]
            va_q = [item_va[int(i)] for i in top20_idx if int(i) in item_va]
            top20_va_per_quadrant.append(np.array(va_q) if va_q else np.zeros((1, 2)))

    # Exclude padding (0) and items with VA=(0,0)
    valid_mask = (item_va_arr[:, 0] != 0) | (item_va_arr[:, 1] != 0)
    results["score_shift"] = score_shift[valid_mask]
    results["item_va_arr"] = item_va_arr[valid_mask]
    results["top20_va_per_quadrant"] = top20_va_per_quadrant

    return results


def pick_example_users(states: dict, n: int = 4) -> list:
    """Select n users with large emotion drift."""
    adm = states["adm"]
    # One representative per quadrant from the top 20% drift users
    thresh = np.percentile(adm, 80)
    high_drift_idx = np.where(adm >= thresh)[0]

    selected = []
    a_n = states["a_n"][high_drift_idx]
    for q, (v_sign, a_sign) in enumerate([(1,1),(-1,1),(-1,-1),(1,-1)]):
        mask = (np.sign(a_n[:, 0]) == v_sign) & (np.sign(a_n[:, 1]) == a_sign)
        cands = high_drift_idx[mask]
        if len(cands) > 0:
            # User with the highest drift in this quadrant
            best = cands[np.argmax(adm[cands])]
            selected.append(int(best))
    return selected[:n]


def plot_figure(states: dict, out_path: str):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("AffSR: Emotion-Aware Recommendation", fontsize=14, fontweight="bold", y=1.02)

    a_n    = states["a_n"]      # (N, 2)
    va_long = states["va_long"] # (N, 2)
    beta   = states["beta"]     # (N, 4)
    topk_va = states["topk_va"] # (N, 2)

    # ── Panel (a): User emotion trajectory in VA space ────────────────
    ax = axes[0]
    ax.set_title("(a) User Emotion States in VA Space", fontsize=11)

    # Background distribution
    ax.scatter(va_long[:, 0], va_long[:, 1], alpha=0.10, s=8,
               color="#4393C3", label="Long-term $\\bar{v}$")
    ax.scatter(a_n[:, 0], a_n[:, 1], alpha=0.10, s=8,
               color="#D6604D", label="Short-term $a_n$")

    # Arrow trajectories for 4 representative users
    example_idxs = pick_example_users(states, n=4)
    for i, idx in enumerate(example_idxs):
        vl = va_long[idx]
        an = a_n[idx]
        ax.annotate("", xy=an, xytext=vl,
                    arrowprops=dict(arrowstyle="->", color=QUADRANT_COLORS[i], lw=2))
        ax.scatter(*vl, color=QUADRANT_COLORS[i], s=60, zorder=5, marker="o")
        ax.scatter(*an, color=QUADRANT_COLORS[i], s=60, zorder=5, marker="*")

    ax.axhline(0, color="gray", lw=0.8, ls="--")
    ax.axvline(0, color="gray", lw=0.8, ls="--")
    ax.set_xlabel("Valence", fontsize=11)
    ax.set_ylabel("Arousal", fontsize=11)
    ax.set_xlim(-1.1, 1.1); ax.set_ylim(-1.1, 1.1)
    ax.text( 0.55,  0.95, "Joy/Excitement", fontsize=7, color="#888888", ha="center")
    ax.text(-0.55,  0.95, "Anger/Fear",     fontsize=7, color="#888888", ha="center")
    ax.text(-0.55, -0.95, "Sadness",        fontsize=7, color="#888888", ha="center")
    ax.text( 0.55, -0.95, "Calm/Relief",    fontsize=7, color="#888888", ha="center")

    legend_handles = [
        mpatches.Patch(color="#4393C3", alpha=0.6, label="Long-term $\\bar{v}$"),
        mpatches.Patch(color="#D6604D", alpha=0.6, label="Short-term $a_n$"),
        plt.Line2D([0],[0], marker="", color="#555555", lw=1.5,
                   label="Drift ($\\rightarrow$)"),
    ]
    ax.legend(handles=legend_handles, fontsize=7, loc="lower right", framealpha=0.9)

    # ── Panel (b): β weights — mean per quadrant ──────────────────────
    ax = axes[1]
    ax.set_title("(b) MoE Expert Activation by Quadrant", fontsize=11)

    # Classify users by short-term sentiment quadrant
    quadrant_beta = []
    labels = []
    for q, (v_sign, a_sign) in enumerate([(1,1),(-1,1),(-1,-1),(1,-1)]):
        mask = (np.sign(a_n[:, 0]) == v_sign) & (np.sign(a_n[:, 1]) == a_sign)
        if mask.sum() > 0:
            quadrant_beta.append(beta[mask].mean(axis=0))
            labels.append(QUADRANT_EMOTIONS[q])

    x = np.arange(4)
    width = 0.2
    for i, (qb, lbl) in enumerate(zip(quadrant_beta, labels)):
        ax.bar(x + i * width, qb, width, label=lbl, color=QUADRANT_COLORS[i], alpha=0.85)

    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(QUADRANT_NAMES, fontsize=9)
    ax.set_ylabel("Mean β weight", fontsize=11)
    ax.set_ylim(0, 1)
    ax.legend(fontsize=7, loc="upper right")
    ax.axhline(0.25, color="gray", lw=0.8, ls="--", label="uniform (0.25)")

    # ── Panel (c): Smooth Heatmap + Centroid Ellipse ──────────────────
    ax = axes[2]
    score_shift      = states.get("score_shift")
    item_va_arr      = states.get("item_va_arr")
    top20_va_per_q   = states.get("top20_va_per_quadrant")

    GRID = 30
    edges = np.linspace(-1, 1, GRID + 1)
    centers = (edges[:-1] + edges[1:]) / 2

    if score_shift is not None and item_va_arr is not None:
        # ── Aggregate into 30×30 grid ────────────────────────────────
        grid_sum   = np.zeros((GRID, GRID))
        grid_count = np.zeros((GRID, GRID))
        vi = np.digitize(item_va_arr[:, 0], edges) - 1  # valence bin
        ai = np.digitize(item_va_arr[:, 1], edges) - 1  # arousal bin
        vi = np.clip(vi, 0, GRID - 1)
        ai = np.clip(ai, 0, GRID - 1)
        for i in range(len(score_shift)):
            grid_sum[ai[i], vi[i]]   += score_shift[i]
            grid_count[ai[i], vi[i]] += 1
        grid_mean = np.where(grid_count > 0, grid_sum / (grid_count + 1e-8), 0.0)

        # ── Gaussian smoothing ───────────────────────────────────────
        grid_smooth = gaussian_filter(grid_mean, sigma=1.5)

        # ── imshow ───────────────────────────────────────────────────
        vabs = np.percentile(np.abs(grid_smooth[grid_smooth != 0]), 95) if (grid_smooth != 0).any() else 1.0
        im = ax.imshow(
            grid_smooth,
            extent=[-1, 1, -1, 1], origin="lower",
            cmap=PASTEL_CMAP, vmin=-vabs, vmax=vabs,
            aspect="auto", interpolation="bilinear",
        )
        plt.colorbar(im, ax=ax, label="Score(Joy) − Score(Sad)", shrink=0.85, pad=0.02)

        # ── 5 contour lines ──────────────────────────────────────────
        V, A = np.meshgrid(centers, centers)
        ax.contour(V, A, grid_smooth, levels=5, colors="white", linewidths=0.6, alpha=0.6)

    ax.axhline(0, color="white", lw=0.7, ls="--", alpha=0.7)
    ax.axvline(0, color="white", lw=0.7, ls="--", alpha=0.7)
    ax.set_xlabel("Item Valence", fontsize=11)
    ax.set_ylabel("Item Arousal", fontsize=11)
    ax.set_xlim(-1, 1); ax.set_ylim(-1, 1)
    ax.set_title("(c) Emotion-Conditioned Item Score Shift", fontsize=11)

    plt.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Figure saved: {out_path}")


def compute_shift_only(model: AffSR, data_dir: str, device: torch.device,
                        item_va: dict, seed: int = 0) -> dict:
    """Compute only score_shift + top20_va_per_quadrant (for comparison figures)."""
    torch.manual_seed(seed)
    ds = AffSRDataset(data_dir, split="test", max_seq_len=50, full_ce=True)
    sample_batch = next(iter(DataLoader(ds, batch_size=32, shuffle=True, num_workers=0)))
    sample_batch = {k: v.to(device) for k, v in sample_batch.items()}

    B32 = sample_batch["item_seq"].size(0)
    N   = model.num_items + 1
    d   = model.d_model

    item_va_arr = np.zeros((N, 2), dtype=np.float32)
    for idx, va in item_va.items():
        if idx < N:
            item_va_arr[idx] = va

    if model.baseline_only:
        # no_moe + baseline: no MoE, penalty only
        centroids_np = np.array([[ 0.83,  0.61],
                                  [-0.53,  0.57],
                                  [ 0.75, -0.48],
                                  [-0.69, -0.43]])
    else:
        centroids_np = model.affdrift.centroids.detach().cpu().numpy()

    joy_idx = int(np.argmax(centroids_np[:, 0] + centroids_np[:, 1]))
    sad_idx = int(np.argmin(centroids_np[:, 0] + centroids_np[:, 1]))
    lambda_mc = torch.nn.functional.softplus(model.lambda_mc).item()

    with torch.no_grad():
        r_u     = model.sasrec(sample_batch["item_seq"], sample_batch["seq_mask"])
        r_bar_u = model._user_repr(r_u, torch.zeros(B32, model.K, device=device))

        def _score(beta_idx, a_n_np):
            a_n_t = torch.tensor(a_n_np, device=device, dtype=torch.float32)
            beta_fixed = torch.zeros(B32, model.K, device=device)
            beta_fixed[:, beta_idx] = 1.0
            chunks = []
            for start in range(0, N, 512):
                end   = min(start + 512, N)
                C     = end - start
                e_id  = model.item_emb(torch.arange(start, end, device=device))
                e_flat = e_id.unsqueeze(0).expand(B32, C, d).reshape(B32*C, d)
                b_exp  = beta_fixed.unsqueeze(1).expand(B32, C, model.K).reshape(B32*C, model.K)
                r_exp  = r_bar_u.unsqueeze(1).expand(B32, C, d).reshape(B32*C, d)
                sc = model._chunk_forward(e_flat, b_exp, r_exp)
                chunks.append(sc.view(B32, C))
            moe_sc = torch.cat(chunks, dim=1).mean(dim=0)
            all_va = torch.tensor(item_va_arr, device=device, dtype=torch.float32)
            penalty = lambda_mc * torch.norm(all_va - a_n_t.unsqueeze(0), dim=-1)
            return (moe_sc - penalty).cpu().numpy()

        s_joy = _score(joy_idx, centroids_np[joy_idx])
        s_sad = _score(sad_idx, centroids_np[sad_idx])
        shift = s_joy - s_sad

        top20 = []
        for q in range(4):
            sc_q = _score(q, centroids_np[q])
            top20_idx = np.argsort(sc_q)[::-1][:20]
            va_q = [item_va[int(i)] for i in top20_idx if int(i) in item_va]
            top20.append(np.array(va_q) if va_q else np.zeros((1, 2)))

    valid = (item_va_arr[:, 0] != 0) | (item_va_arr[:, 1] != 0)
    return {
        "score_shift":            shift[valid],
        "item_va_arr":            item_va_arr[valid],
        "top20_va_per_quadrant":  top20,
        "centroids":              centroids_np,
        "joy_idx":                joy_idx,
        "sad_idx":                sad_idx,
    }


def _draw_shift_panel(ax, data: dict, title: str, show_cbar: bool = True,
                      shared_vabs: float = None):
    """Draw Score Shift heatmap + Ellipse on ax. Reusable."""
    score_shift  = data["score_shift"]
    item_va_arr  = data["item_va_arr"]
    top20_va_q   = data["top20_va_per_quadrant"]

    GRID = 30
    edges   = np.linspace(-1, 1, GRID + 1)
    centers = (edges[:-1] + edges[1:]) / 2

    grid_sum   = np.zeros((GRID, GRID))
    grid_count = np.zeros((GRID, GRID))
    vi = np.clip(np.digitize(item_va_arr[:, 0], edges) - 1, 0, GRID - 1)
    ai = np.clip(np.digitize(item_va_arr[:, 1], edges) - 1, 0, GRID - 1)
    for i in range(len(score_shift)):
        grid_sum[ai[i], vi[i]]   += score_shift[i]
        grid_count[ai[i], vi[i]] += 1
    grid_mean   = np.where(grid_count > 0, grid_sum / (grid_count + 1e-8), 0.0)
    grid_smooth = gaussian_filter(grid_mean, sigma=1.5)

    vabs = shared_vabs if shared_vabs else (
        np.percentile(np.abs(grid_smooth[grid_smooth != 0]), 95)
        if (grid_smooth != 0).any() else 1.0
    )

    im = ax.imshow(grid_smooth, extent=[-1, 1, -1, 1], origin="lower",
                   cmap=PASTEL_CMAP, vmin=-vabs, vmax=vabs,
                   aspect="auto", interpolation="bilinear")
    if show_cbar:
        plt.colorbar(im, ax=ax, label="Score(Joy) − Score(Sad)", shrink=0.85, pad=0.02)

    V, A = np.meshgrid(centers, centers)
    ax.contour(V, A, grid_smooth, levels=5, colors="white", linewidths=0.6, alpha=0.6)

    ax.axhline(0, color="white", lw=0.7, ls="--", alpha=0.7)
    ax.axvline(0, color="white", lw=0.7, ls="--", alpha=0.7)
    ax.set_xlabel("Item Valence", fontsize=11)
    ax.set_ylabel("Item Arousal", fontsize=11)
    ax.set_title(title, fontsize=11)
    return im, vabs


def plot_moe_comparison(data_full: dict, data_nomoe: dict, out_path: str):
    """3-panel comparison figure: w/o MoE | AffSR full | MoE contribution."""
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    fig.suptitle("Effect of Emotion MoE on Item Score Shift\n"
                 "(red = Joy-favored,  blue = Sad-favored)",
                 fontsize=13, fontweight="bold")

    # Panels (a)(b) share the same color scale
    v1 = np.percentile(np.abs(data_full["score_shift"]), 95)
    v2 = np.percentile(np.abs(data_nomoe["score_shift"]), 95)
    shared_vabs = max(v1, v2)

    im, _ = _draw_shift_panel(axes[0], data_nomoe,
                               title="(a) w/o MoE  (penalty only)",
                               show_cbar=False, shared_vabs=shared_vabs)
    im, _ = _draw_shift_panel(axes[1], data_full,
                               title="(b) AffSR (w/ MoE)",
                               show_cbar=False, shared_vabs=shared_vabs)

    # ── Panel (c): Net MoE contribution = full − nomoe ───────────────
    moe_contribution = data_full["score_shift"] - data_nomoe["score_shift"]
    data_moe_only = {
        "score_shift":           moe_contribution,
        "item_va_arr":           data_full["item_va_arr"],
        "top20_va_per_quadrant": data_full["top20_va_per_quadrant"],
    }
    im_c, vabs_c = _draw_shift_panel(axes[2], data_moe_only,
                                      title="(c) MoE contribution  [(b) − (a)]",
                                      show_cbar=False)

    # Shared colorbar for (a)(b)
    fig.subplots_adjust(right=0.88, wspace=0.35)
    cbar_ax1 = fig.add_axes([0.60, 0.12, 0.015, 0.72])
    fig.colorbar(im, cax=cbar_ax1, label="Score(Joy) − Score(Sad)")

    # Separate colorbar for (c)
    cbar_ax2 = fig.add_axes([0.93, 0.12, 0.015, 0.72])
    fig.colorbar(im_c, cax=cbar_ax2, label="MoE Δ score")

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Comparison figure saved: {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt",       default="outputs/v10_final/affsr_full_movies/best.pt")
    parser.add_argument("--nomoe_ckpt", default="outputs/v10_final/ablation_no_moe_movies/best.pt")
    parser.add_argument("--data_dir",   default="data/processed/movies_tv_2021_2023")
    parser.add_argument("--dataset",    default="movies")
    parser.add_argument("--out_dir",    default="outputs/figures")
    parser.add_argument("--n_users",    type=int, default=2000)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ts = datetime.now().strftime("%m%d_%H%M")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Device: {device}  |  timestamp: {ts}")

    def save(fig_fn, stem):
        """Save figure as both PDF and PNG with timestamp in filename."""
        for ext in ("pdf", "png"):
            path = str(out_dir / f"{stem}_{ts}.{ext}")
            fig_fn(path)

    # ── 3-panel emotion viz ──────────────────────────────────────────
    print("Loading AffSR full model...")
    model_full, _ = load_model(args.ckpt, args.data_dir, device)

    print(f"Extracting states for {args.n_users} users...")
    states = extract_user_states(model_full, args.data_dir, device, n_users=args.n_users)
    adm = states["adm"]
    print(f"  ADM mean={adm.mean():.4f}, max={adm.max():.4f}")
    print(f"  Users with drift > 0.3: {(adm > 0.3).sum()}")

    save(lambda p: plot_figure(states, p), "affsr_emotion_viz")

    # ── MoE 3-panel comparison figure ────────────────────────────────
    if Path(args.nomoe_ckpt).exists():
        print("\nLoading w/o MoE model...")
        model_nomoe, _ = load_model(args.nomoe_ckpt, args.data_dir, device, no_moe=True)
        with open(Path(args.data_dir) / "item_va.json") as f:
            raw_va = json.load(f)
        item_va_dict = {int(k): v["va"] for k, v in raw_va.items() if k != "__meta__"}

        print("Computing score shift for w/o MoE...")
        data_nomoe = compute_shift_only(model_nomoe, args.data_dir, device, item_va_dict)

        save(lambda p: plot_moe_comparison(states, data_nomoe, p), "affsr_moe_comparison")
    else:
        print(f"[SKIP] no_moe ckpt not found: {args.nomoe_ckpt}")


if __name__ == "__main__":
    main()
