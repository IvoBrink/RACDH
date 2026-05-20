import os
import sys
import math
import numpy as np
import torch
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
from RACDH.config import params
from RACDH.data_generation.utils.reading_data import load_json

from sklearn.decomposition import PCA

# -----------------------
# Config
# -----------------------
MODELS = [
    "Llama-3.1-8B",
    "Mistral-7B-v0.1",
    "Qwen2.5-7B",
]

REP_KEY = "first_token_generation"
SEED = 0
N_COMPONENTS = 2

# Plot settings
FIGSIZE_PER_AX = (2.2, 2.2)
POINT_SIZE = 2.0
POINT_ALPHA = 0.6

L2_NORMALIZE = True

# Use all points (set e.g. 20000 if needed)
N_POINTS_PER_MODEL = None

# You can change this freely (8, 10, 12, ...)
N_COLS_WRAP = 4

OUT_DIR = os.path.join(params.output_path, "plots")


# -----------------------
# Helpers
# -----------------------
def stratified_indices(labels, n, seed=0):
    rng = np.random.default_rng(seed)
    labels = np.asarray(labels)
    N = len(labels)

    if n is None or n >= N:
        return np.arange(N)

    idx0 = np.where(labels == 0)[0]
    idx1 = np.where(labels == 1)[0]

    n0 = min(n // 2, len(idx0))
    n1 = min(n - n0, len(idx1))

    pick0 = rng.choice(idx0, size=n0, replace=False)
    pick1 = rng.choice(idx1, size=n1, replace=False)

    idx = np.concatenate([pick0, pick1])
    rng.shuffle(idx)
    return idx


def l2_normalize_np(X, eps=1e-12):
    return X / (np.linalg.norm(X, axis=1, keepdims=True) + eps)


def compute_layer_xy_per_layer_pca(hidden_states, indices, rep_key, normalize=True):
    """
    Compute PCA for ALL layers except layer 0.
    """
    L, D = hidden_states[0][rep_key].shape
    layers_to_use = list(range(1, L))  # EXCLUDE layer 0
    n_plot = len(indices)

    print(f"Using layers {layers_to_use[0]}..{layers_to_use[-1]} (total {len(layers_to_use)})")

    layer_xy = []
    for l in layers_to_use:
        X = torch.stack([hidden_states[i][rep_key][l] for i in indices], dim=0)
        X = X.detach().cpu().float().numpy()

        if normalize:
            X = l2_normalize_np(X)

        pca = PCA(n_components=N_COMPONENTS, random_state=SEED, svd_solver="randomized")
        layer_xy.append(pca.fit_transform(X).astype(np.float32))

    return layer_xy, layers_to_use


def plot_all_layers_vertical_grid(layer_xy, labels, layer_ids, model_name, n_cols_wrap):
    """
    Plot all layers in a wrapped grid, hiding unused panels in the final row.
    Works for ANY n_cols_wrap.
    """
    labels = np.asarray(labels)
    mask_context = labels == 0
    mask_param = labels == 1

    c_context = "#61b8ff"
    c_param = "#f4a0b7"

    L = len(layer_xy)
    if L == 0:
        raise ValueError("No layers to plot (layer_xy is empty).")

    n_cols = min(n_cols_wrap, L)
    n_rows = int(math.ceil(L / n_cols))  # <-- FIX: ceil, not //

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(FIGSIZE_PER_AX[0] * n_cols, FIGSIZE_PER_AX[1] * n_rows),
        squeeze=False
    )

    # Plot layers
    for i, (XY, layer_id) in enumerate(zip(layer_xy, layer_ids)):
        r, c = divmod(i, n_cols)
        ax = axes[r][c]

        ax.scatter(XY[mask_context, 0], XY[mask_context, 1],
                   s=POINT_SIZE, alpha=POINT_ALPHA, c=c_context, linewidths=0)
        ax.scatter(XY[mask_param, 0], XY[mask_param, 1],
                   s=POINT_SIZE, alpha=POINT_ALPHA, c=c_param, linewidths=0)

        ax.set_title(f"Layer {layer_id}", fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])

        for spine in ax.spines.values():
            spine.set_linewidth(0.6)
            spine.set_alpha(0.6)

    # Turn off any unused axes (last row will have blanks if not exact multiple)
    total_axes = n_rows * n_cols
    for j in range(L, total_axes):
        r, c = divmod(j, n_cols)
        axes[r][c].axis("off")

    # Legend once per figure
    from matplotlib.lines import Line2D
    fig.legend(
        handles=[
            Line2D([0], [0], marker='o', color='w', label='Contextual',
                   markerfacecolor=c_context, markersize=8),
            Line2D([0], [0], marker='o', color='w', label='Parametric',
                   markerfacecolor=c_param, markersize=8),
        ],
        loc="lower center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, 0.01),
        fontsize=11
    )

    fig.suptitle(
        f"Per-layer PCA of Hidden State Activations — {model_name} (layers {layer_ids[0]}–{layer_ids[-1]})",
        fontsize=14
    )
    fig.tight_layout(rect=(0, 0.05, 1, 0.95))
    return fig


# -----------------------
# Main
# -----------------------
def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    for MODEL in MODELS:
        print(f"\n=== {MODEL} ===")

        hs_path = os.path.join(params.output_path, f"{MODEL}/gpt-4o-mini/hiddens_all_2.pt")
        meta_rel = f"{MODEL}/gpt-4o-mini/hiddens_metadata_all_2.json"

        hidden_states = torch.load(hs_path, map_location="cpu")
        meta = load_json(meta_rel)

        labels_all = np.array(
            [1 if m.get("label") == "parametric" else 0 for m in meta],
            dtype=np.int64
        )

        plot_idx = stratified_indices(labels_all, N_POINTS_PER_MODEL, SEED)
        labels_plot = labels_all[plot_idx]

        layer_xy, layer_ids = compute_layer_xy_per_layer_pca(
            hidden_states, plot_idx, REP_KEY, L2_NORMALIZE
        )

        fig = plot_all_layers_vertical_grid(
            layer_xy=layer_xy,
            labels=labels_plot,
            layer_ids=layer_ids,
            model_name=MODEL,
            n_cols_wrap=N_COLS_WRAP,
        )

        out_path = os.path.join(
            OUT_DIR,
            f"layer_pca_{MODEL}_{REP_KEY}_layers{layer_ids[0]}to{layer_ids[-1]}_wrap{N_COLS_WRAP}.png"
        )
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        print(f"Saved → {out_path}")

        plt.show()
        plt.close(fig)


if __name__ == "__main__":
    main()
