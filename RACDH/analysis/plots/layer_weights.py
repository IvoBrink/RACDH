#!/usr/bin/env python3
"""
Density‑curve plot of layer‑aggregation weights with improved aesthetics.
"""

from __future__ import annotations

import os, sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from scipy.ndimage import gaussian_filter1d


sys.path.append(os.path.abspath("/home/ibrink/RACDH/RACDH/"))
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
    "first_token_entity"     : "First‑token entity",
    "last_token_entity"      : "Last‑token entity",
    "first_token_generation" : "First‑token generation",
}


HEADROOM = 0.08  # 8 % extra space above tallest curve


def _smooth_curve(x: np.ndarray, y: np.ndarray, *, n: int = 600,
                  sigma: float = 3.5) -> tuple[np.ndarray, np.ndarray]:
    """Return smoothed y on a dense x‑grid using Gaussian blur."""
    x_dense = np.linspace(x.min(), x.max(), n)
    y_dense = np.interp(x_dense, x, y)
    y_smooth = gaussian_filter1d(y_dense, sigma=sigma, mode="nearest")
    return x_dense, y_smooth


def _light_rgba(hex_colour: str, alpha: float = 0.15):
    """Blend the hex colour toward white, retaining *alpha* opacity."""
    rgb = np.array(mcolors.to_rgb(hex_colour))
    lighter = rgb * 0.55 + 0.45
    return (*lighter, alpha)

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    classifier = "logreg"
    models  = ["Llama-3.1-8B", "Mistral-7B-v0.1"]
    tokens  = [
        "first_token_entity",
        "last_token_entity",
        "first_token_generation",
    ]

    weights = {
        (m, t): np.load(f"RACDH/data/plots/{m}/layer_weights_{classifier}_{t}.npy")
        for m in models for t in tokens
    }

    n_layers = next(iter(weights.values())).shape[0]
    layers   = np.arange(n_layers, dtype=float)


    fig_h = 4.0 * len(models)
    fig_w = 0.23 * n_layers + 5
    fig, axes = plt.subplots(
        nrows=len(models), figsize=(fig_w, fig_h),
        sharex=True, sharey=True, constrained_layout=True,
    )
    axes = np.atleast_1d(axes)

    for ax, model in zip(axes, models):
        tallest = 0.0
        ax.set_axisbelow(False) 
        ax.grid(zorder=1)

        for token in tokens:
            w = weights[(model, token)]
            x, y = _smooth_curve(layers, w, sigma=7)
            tallest = max(tallest, float(y.max()))

            ax.fill_between(
                x, 0, y,
                color=_light_rgba(PALETTE[token]),
                zorder=2,
            )
            
            ax.plot(
                x, y,
                color=PALETTE[token],
                linewidth=2.8,
                zorder=3,
                label=TOKEN_LABEL[token] if ax is axes[0] else None,
            )

        y_max = tallest * (1 + HEADROOM)
        ax.set_ylim(0, y_max)
        ax.set_xticks(range(0, n_layers, 4))
        ax.set_title(model, loc="left", pad=10)
        ax.set_ylabel(
            "Aggregation\nweight", rotation=0,
            ha="right", va="center", labelpad=34,
        )

    axes[-1].set_xlabel("Transformer layer index")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="upper center",
        bbox_to_anchor=(0.5, 1.09), ncol=len(tokens), frameon=False,
    )
    fig.suptitle(
        f"Layer‑wise aggregation weights for {classifier}",
        y=1.15, fontsize=16,
    )


    fig.savefig(
        params.output_path + f"/plots/layer_weight_density_grid_{classifier}.png",
        dpi=300, bbox_inches="tight",
    )
 
if __name__ == "__main__":
    main()
