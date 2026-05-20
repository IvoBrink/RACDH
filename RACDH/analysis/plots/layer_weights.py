#!/usr/bin/env python3
"""
Density-curve plot of layer-aggregation weights
(compact, cropped x-range, no title).
"""

from __future__ import annotations

import os, sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from scipy.ndimage import gaussian_filter1d

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
from RACDH.config import params


plt.rcParams.update({
    "font.size": 13,
    "axes.titlesize": 14,
    "axes.labelsize": 13,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 12,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.linestyle": ":",
    "grid.alpha": 0.25,
})

PALETTE = {
    "first_token_entity"     : "#6630ff",   # purple
    "last_token_entity"      : "#00b8f9",   # teal
    "first_token_generation" : "#0a2b87",   # navy
}
TOKEN_LABEL = {
    "first_token_entity"     : "First-token entity",
    "last_token_entity"      : "Last-token entity",
    "first_token_generation" : "First-token generation",
}

HEADROOM = 0.08  # 8% extra space above tallest curve


def _smooth_curve(x: np.ndarray, y: np.ndarray, *, n: int = 600,
                  sigma: float = 3.5) -> tuple[np.ndarray, np.ndarray]:
    """Return smoothed y on a dense x-grid using Gaussian blur."""
    x_dense = np.linspace(x.min(), x.max(), n)
    y_dense = np.interp(x_dense, x, y)
    y_smooth = gaussian_filter1d(y_dense, sigma=sigma, mode="nearest")
    return x_dense, y_smooth


def _light_rgba(hex_colour: str, alpha: float = 0.15):
    """Blend the hex colour toward white, retaining *alpha* opacity."""
    rgb = np.array(mcolors.to_rgb(hex_colour))
    lighter = rgb * 0.55 + 0.45
    return (*lighter, alpha)


def main() -> None:
    classifier = "logreg"
    models  = ["Llama-3.1-8B", "Mistral-7B-v0.1", "Qwen2.5-7B"]
    tokens  = [
        "last_token_entity",
        "first_token_generation",
    ]

    # Load weights (allow different lengths per model)
    weights = {}
    n_layers_by_model = {}

    for m in models:
        for t in tokens:
            path = f"RACDH/data/plots/{m}/layer_weights_{classifier}_{t}.npy"
            w = np.load(path)
            w = np.asarray(w).reshape(-1)
            weights[(m, t)] = w
            n_layers_by_model[m] = max(n_layers_by_model.get(m, 0), w.shape[0])

    # --- Crop uninformative layers ---
    x_min = 12
    x_max = 24   # <<< capped here

    # --- FIGURE SIZE: slightly taller but still compact ---
    fig_h_per_row = 2
    fig_h = fig_h_per_row * len(models)
    fig_w = 0.22 * (x_max - x_min + 1) + 5.0

    fig, axes = plt.subplots(
        nrows=len(models),
        figsize=(fig_w, fig_h),
        sharex=True,
        sharey=True,
        constrained_layout=True,
        gridspec_kw={"hspace": 0.18},
    )
    axes = np.atleast_1d(axes)

    for ax, model in zip(axes, models):
        tallest = 0.0
        ax.set_axisbelow(False)
        ax.grid(zorder=1)

        n_layers = n_layers_by_model[model]
        layers = np.arange(n_layers, dtype=float)

        for token in tokens:
            w = weights[(model, token)]
            n = min(len(w), len(layers))
            w_use = w[:n]
            layers_use = layers[:n]

            x, y = _smooth_curve(layers_use, w_use, sigma=7)
            tallest = max(tallest, float(y.max()))

            ax.fill_between(
                x, 0, y,
                color=_light_rgba(PALETTE[token]),
                zorder=2,
            )
            ax.plot(
                x, y,
                color=PALETTE[token],
                linewidth=2.4,
                zorder=3,
                label=TOKEN_LABEL[token] if ax is axes[0] else None,
            )

        y_max = tallest * (1 + HEADROOM) if tallest > 0 else 1.0
        ax.set_ylim(0, y_max)

        # Model label
        ax.set_title(model, loc="left", pad=4, fontsize=14)

    # Shared y-label
    fig.supylabel("Aggregation weight", x=-0.03, fontsize=16)

    # Global x-axis settings (cropped)
    axes[-1].set_xlabel(
        "Transformer layer index",
        fontsize=16
    )
    for ax in axes:
        ax.set_xlim(x_min, x_max)
        ax.set_xticks(range(x_min, x_max + 1, 2))

    # Legend at top, no title
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc="upper center",
        bbox_to_anchor=(0.6, 1.03),
        ncol=len(tokens),
        frameon=False,
        handlelength=2.2,
        columnspacing=1.8,
        fontsize=14,
    )

    fig.savefig(
        params.output_path + f"/plots/layer_weight_density_grid_{classifier}.png",
        dpi=300,
        bbox_inches="tight",
    )


if __name__ == "__main__":
    main()
