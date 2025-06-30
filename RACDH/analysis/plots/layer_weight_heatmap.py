#!/usr/bin/env python3
"""
Heat-map of layer-aggregation weights (12-column version).

• Col-0  : sum of layers  0-11
• Col-1…10 : individual layers 12-21
• Col-11 : sum of layers 22-(31|32)
"""

from __future__ import annotations
import os, sys, numpy as np, numpy.ma as ma
import matplotlib.pyplot as plt
# ---------------------------------------------------------------------------
sys.path.append(os.path.abspath("/home/ibrink/RACDH/RACDH/"))
from RACDH.config import params                         # noqa: E402
# ---------------------------------------------------------------------------

# ── NEW COMPRESSION FUNCTION ───────────────────────────────────────────────
def _compress_layers(w: np.ndarray) -> np.ndarray:
    """Return 12-value vector with the requested collapses."""
    n = w.shape[0]
    if n not in {32, 33}:
        raise ValueError(f"Expected 32 or 33 layers, got {n}.")
    left  = w[0:12].sum()          # 0-11
    mid   = w[12:22]               # 12-21  (10 layers)
    right = w[22:].sum()           # 22-31/32
    return np.concatenate(([left], mid, [right]))   # (12,)

# ── MAIN (unchanged except for labels/shape tweaks) ───────────────────────
def main() -> None:
    classifier = "logreg"
    models  = ["Llama-3.1-8B", "Mistral-7B-v0.1"]
    tokens  = ["first_token_entity", "last_token_entity", "first_token_generation"]
    token_lbl = {
        "first_token_entity":      "First-token entity",
        "last_token_entity":       "Last-token entity",
        "first_token_generation":  "First-token generation",
    }

    rows, labels = [], []
    for i, model in enumerate(models):
        for t in tokens:
            w = np.load(f"RACDH/data/plots/{model}/layer_weights_{classifier}_{t}.npy")
            rows.append(_compress_layers(w))
            labels.append(f"{token_lbl[t]}  ({model})")
        if i < len(models) - 1:                    # spacer row
            rows.append(np.full(12, np.nan)); labels.append("")

    data = ma.masked_invalid(np.vstack(rows))      # (rows, 12)

    # ――― Plot ―――
    fig_h = 0.6 * data.shape[0] + 2
    fig_w = 0.7 * data.shape[1] + 3
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), constrained_layout=True)

    cmap = plt.cm.Blues.copy();
    im = ax.imshow(data, cmap=cmap, aspect="auto", vmin=0, vmax=data.max())

    # annotate every valid cell
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            if not data.mask[i, j]:
                ax.text(j, i, f"{data[i, j]:.3f}", ha="center", va="center",
                        fontsize=7)

    # ticks & labels
    last_idx = rows[0].size + 10   # could also derive from w.shape[0]-1
    cols = ["0-11"] + [str(k) for k in range(12, 22)] + [f"22-{(32 if rows[0].size==12 else 31)}"]
    ax.set_xticks(range(12));  ax.set_xticklabels(cols, rotation=45, ha="right")
    ax.set_yticks(range(len(labels)));  ax.set_yticklabels(labels)
    ax.set_xlabel("Transformer layer index (collapsed)")
    ax.set_title(f"Layer-aggregation weights – {classifier}", pad=12)

    # grid
    ax.set_xticks(np.arange(-.5, 12, 1), minor=True)
    ax.set_yticks(np.arange(-.5, len(labels), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.2)
    ax.tick_params(which="minor", bottom=False, left=False)

    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02, label="Weight")
    fig.savefig(params.output_path + f"/plots/layer_weight_heatmap_{classifier}.png",
                dpi=300, bbox_inches="tight")

if __name__ == "__main__":
    main()
