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
    "Qwen2.5-7B"
]
REP_KEY = "first_token_generation"

SEED = 0
N_COMPONENTS = 2

# Plot exactly N_LAYERS_TO_PLOT columns for all models (aligned)
N_LAYERS_TO_PLOT = 8

# Layout for the combined figure:
# rows = number of models, cols = N_LAYERS_TO_PLOT
FIGSIZE_PER_AX = (2.0, 2.0)
POINT_SIZE = 2.0
POINT_ALPHA = 0.6

# Normalization
L2_NORMALIZE = True

# Choose how to pick the 8 layers (global, shared across all models)
LAYER_PICK_MODE = "custom"  # {"even", "last", "first", "custom"}
# CUSTOM_LAYERS = [4, 8, 12, 16, 20, 24, 28, 32]  # global columns
CUSTOM_LAYERS = np.linspace(0, 32, num=8).astype(int).tolist()  # alternative


def stratified_indices(labels, n, seed=0):
    """(Unused by default) If you later want balanced subsampling."""
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


def l2_normalize_np(X: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    return X / (norms + eps)


def compute_layer_xy_per_layer_pca(hidden_states, indices, rep_key, normalize=True):
    """
    For each layer l:
      - gather vectors X_l = hidden_states[i][rep_key][l] for i in indices
      - (optional) L2 normalize rows
      - fit PCA on X_l
      - transform X_l to 2D

    Returns:
      layer_xy: list length L, each array [n_plot, 2]
    """
    L, D = hidden_states[0][rep_key].shape
    n_plot = len(indices)

    layer_xy = []
    for l in range(L):
        X = torch.stack([hidden_states[i][rep_key][l] for i in indices], dim=0)
        X = X.detach().cpu().float().numpy()

        if normalize:
            X = l2_normalize_np(X)

        pca = PCA(n_components=N_COMPONENTS, svd_solver="randomized", random_state=SEED)
        XY = pca.fit_transform(X)  # [n_plot, 2]
        layer_xy.append(XY.astype(np.float32))

        if l == 0:
            print(f"Per-layer PCA: L={L}, D={D}, n_plot={n_plot}")

    return layer_xy


def pick_n_layers(L: int, n: int = 8, mode: str = "even", custom=None):
    """
    Return exactly n unique layer indices in [0, L-1].
    If mode='custom', the provided list defines the columns.
    """
    if n <= 0 or L <= 0:
        return []
    n = min(n, L)

    if mode == "even":
        layers = np.linspace(0, L - 1, num=n)
        layers = np.unique(np.round(layers).astype(int)).tolist()

        # If rounding caused duplicates, fill deterministically with missing layers
        if len(layers) < n:
            missing = [x for x in range(L) if x not in set(layers)]
            need = n - len(layers)
            layers += missing[:need]

        return sorted(layers[:n])

    if mode == "last":
        return list(range(L - n, L))

    if mode == "first":
        return list(range(0, n))

    if mode == "custom":
        if custom is None:
            raise ValueError("mode='custom' requires custom list.")
        # NOTE: for aligned plots we do NOT filter here; we want columns shared.
        # Missing layers will be handled at plotting time by leaving blank axes.
        if len(set(custom)) != len(custom):
            raise ValueError("Custom layers must be unique.")
        if len(custom) != n:
            raise ValueError(f"Custom layers must contain exactly {n} layers. Got {len(custom)}.")
        return list(custom)

    raise ValueError(f"Unknown mode: {mode}")


def plot_model_layers_into_axes(
    axes_row,
    layer_xy,
    labels,
    layers_to_plot,
    model_name,
    s=2.0,
    alpha=0.6,
):
    """
    Plot the selected layers for a single model into a single row of axes.
    If a layer doesn't exist for this model, leave that subplot blank (white space).
    """
    labels = np.asarray(labels)
    mask_context = (labels == 0)
    mask_param = (labels == 1)

    c_context = "#61b8ff"
    c_param = "#f4a0b7"

    L = len(layer_xy)

    for col, layer in enumerate(layers_to_plot):
        ax = axes_row[col]

        # If this model doesn't have that layer, leave whitespace
        if layer >= L:
            ax.axis("off")
            continue

        XY = layer_xy[layer]

        ax.scatter(XY[mask_context, 0], XY[mask_context, 1],
                   s=s, alpha=alpha, linewidths=0, c=c_context)
        ax.scatter(XY[mask_param, 0],   XY[mask_param, 1],
                   s=s, alpha=alpha, linewidths=0, c=c_param)

        ax.set_xticks([])
        ax.set_yticks([])

        ax.set_title(f"Layer {layer}", fontsize=12)

        for spine in ax.spines.values():
            spine.set_linewidth(0.6)
            spine.set_alpha(0.6)

    # Put model label on the left side of the row
    axes_row[0].set_ylabel(model_name, fontsize=12, rotation=90)


def main():
    n_models = len(MODELS)
    n_cols = N_LAYERS_TO_PLOT

    fig, axes = plt.subplots(
        n_models,
        n_cols,
        figsize=(FIGSIZE_PER_AX[0] * n_cols, FIGSIZE_PER_AX[1] * n_models),
        squeeze=False,
    )

    # Pick global, shared layers (so columns align across models)
    # For "custom", this is exactly CUSTOM_LAYERS.
    # For others, we base it on the maximum depth across models.
    if LAYER_PICK_MODE == "custom":
        layers_to_plot_global = pick_n_layers(
            L=10**9,  # dummy; unused in custom mode
            n=N_LAYERS_TO_PLOT,
            mode="custom",
            custom=CUSTOM_LAYERS,
        )
    else:
        # Find max L across models (loads each once)
        max_L = 0
        for MODEL in MODELS:
            hs_path = os.path.join(params.output_path, f"{MODEL}/gpt-4o-mini/hiddens_all_2.pt")
            hidden_states = torch.load(hs_path)
            L = hidden_states[0][REP_KEY].shape[0]
            max_L = max(max_L, L)
        layers_to_plot_global = pick_n_layers(
            L=max_L,
            n=N_LAYERS_TO_PLOT,
            mode=LAYER_PICK_MODE,
            custom=None,
        )

    print(f"Global layers (columns): {layers_to_plot_global}")

    for row, MODEL in enumerate(MODELS):
        print(f"\nProcessing model: {MODEL}")

        hs_path = os.path.join(params.output_path, f"{MODEL}/gpt-4o-mini/hiddens_all_2.pt")
        meta_path = f"{MODEL}/gpt-4o-mini/hiddens_metadata_all_2.json"

        hidden_states = torch.load(hs_path)
        meta = load_json(meta_path)

        # 0=contextual, 1=parametric
        labels_all = np.array([1 if m["label"] == "parametric" else 0 for m in meta], dtype=np.int64)

        # If you want balanced subsampling later, replace with:
        # plot_idx = stratified_indices(labels_all, n=20000, seed=SEED)
        plot_idx = np.arange(len(labels_all))
        labels_plot = labels_all  # because plot_idx is full range

        layer_xy_plot = compute_layer_xy_per_layer_pca(
            hidden_states=hidden_states,
            indices=plot_idx,
            rep_key=REP_KEY,
            normalize=L2_NORMALIZE,
        )

        L_model = len(layer_xy_plot)
        missing = [l for l in layers_to_plot_global if l >= L_model]
        if missing:
            print(f"{MODEL}: missing layers (will be blank): {missing}")

        plot_model_layers_into_axes(
            axes_row=axes[row],
            layer_xy=layer_xy_plot,
            labels=labels_plot,
            layers_to_plot=layers_to_plot_global,
            model_name=MODEL,
            s=POINT_SIZE,
            alpha=POINT_ALPHA,
        )

    # Single legend for the whole figure
    from matplotlib.lines import Line2D
    legend_elems = [
        Line2D([0], [0], marker='o', color='w', label='Contextual', markerfacecolor="#61b8ff", markersize=10),
        Line2D([0], [0], marker='o', color='w', label='Parametric', markerfacecolor="#f4a0b7", markersize=10),
    ]
    fig.legend(
        handles=legend_elems,
        loc="lower center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, 0.01),
        fontsize=13,
        handletextpad=0.8,
        columnspacing=1.6,
    )

    fig.suptitle("Per-layer PCA of Hidden State activations", fontsize=14)
    fig.tight_layout(rect=(0, 0.05, 1, 0.95))

    out_path = os.path.join(
        params.output_path,
        "plots",
        f"layer_linearity_all_models_{REP_KEY}_n{N_LAYERS_TO_PLOT}.png",
    )
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"\nSaved combined figure to: {out_path}")

    plt.show()


if __name__ == "__main__":
    main()
