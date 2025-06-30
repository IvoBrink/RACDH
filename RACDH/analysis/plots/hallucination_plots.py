#!/usr/bin/env python3
"""Probability‑density comparison on *normal* vs. *context‑switched* SQuAD.

Two side‑by‑side panels show the distribution of **parametric knowledge
probability** (``p_parametric``) for correct vs. incorrect answers.

Publication‑ready tweaks
────────────────────────
* No whitespace ballooning: x‑limits clamped to ``[0,1]`` and y‑limits start at
  zero with a shared maximum across both panels.
* *Mismatch* regions are shaded grey (normal: ``x>0.5``; context‑switched:
  ``x<0.5``) with centred labels placed in **axes‑relative coordinates** so they
  stay anchored even if y‑limits change.
* KDE negative artefacts clipped to zero, ensuring curves sit exactly on the
  axes without floating.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d

# ── RACDH utilities ─────────────────────────────────────────────────────────
sys.path.append(os.path.abspath("/home/ibrink/RACDH/RACDH/"))
from RACDH.config import params  # noqa: E402

# ── Matplotlib global style tweaks ─────────────────────────────────────────
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
    # Grid: slightly dark grey dots, drawn above shaded regions so always visible
    "grid.color": "0.3",     # black‑ish grey
    "grid.linestyle": ":",
    "grid.linewidth": 0.6,
    "grid.alpha": 0.6,
    })

# ── Helper functions ────────────────────────────────────────────────────────

def apply_kpmg_theme() -> None:  # noqa: D401
    """Apply KPMG corporate palette if available; otherwise fallback."""
    try:
        from RACDH.plotting import apply_kpmg_theme as _kpmg  # type: ignore
        _kpmg()
    except Exception:
        plt.style.use("default")


def kde_line(
    ax: plt.Axes,
    values: pd.Series | np.ndarray,
    label: str,
    colour: str,
    *,
    bins: int = 200,
    sigma: float = 3.0,
) -> None:
    """Plot a histogram‑based KDE with translucent fill, clipped ≥ 0."""
    v = np.asarray(values, dtype=float)
    if v.size == 0:
        return

    hist, edges = np.histogram(v, bins=bins, range=(0, 1), density=True)
    x = 0.5 * (edges[:-1] + edges[1:])
    y = gaussian_filter1d(hist, sigma=sigma)
    y = np.clip(y, 0.0, None)  # remove tiny negative artefacts at edges

    ax.plot(x, y, color=colour, linewidth=2.5, label=label)
    ax.fill_between(x, y, alpha=0.20, color=colour)


# ── Main routine ───────────────────────────────────────────────────────────

def create_plot(model: str, classifier: str, *, prior_mode: bool = False) -> None:
    apply_kpmg_theme()

    colours = (
        {"correct": "#00b8f9", "incorrect": "#00318d"}
        if prior_mode
        else {"correct": "#66d575", "incorrect": "#f24822"}
    )
    shade_colour = "0.92"  # light grey

    # ── Load data ─────────────────────────────────────────────────────────--
    base = Path(f"RACDH/data/plots/{model}")
    with (base / f"results_squad_{classifier}__parametric.json").open("r", encoding="utf-8") as f:
        data_p = json.load(f)
    with (base / f"results_squad_{classifier}__normal.json").open("r", encoding="utf-8") as f:
        data_n = json.load(f)

    df_p = pd.DataFrame(data_p)
    df_n = pd.DataFrame(data_n)

    # Normalise schema ------------------------------------------------------
    for df in (df_p, df_n):
        if "p_parametric" not in df.columns:
            if "p_contextual" in df.columns:
                df.rename(columns={"p_contextual": "p_parametric"}, inplace=True)
            else:
                raise KeyError("Missing parametric probability column.")
        if "answer_correct" not in df.columns:
            raise KeyError("Missing 'answer_correct' label.")

    # ── Figure -------------------------------------------------------------
    fig, (ax_n, ax_p) = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    for ax in (ax_n, ax_p):
        ax.set_xlim(0, 1)

    # Plot Normal ----------------------------------------------------------
    kde_line(
        ax_n,
        df_n.loc[df_n.answer_correct, "p_parametric"],
        f"Correct (n={df_n.answer_correct.sum():,})",
        colours["correct"],
    )
    kde_line(
        ax_n,
        df_n.loc[~df_n.answer_correct, "p_parametric"],
        f"Incorrect (n={(~df_n.answer_correct).sum():,})",
        colours["incorrect"],
    )
    ax_n.axvline(0.5, color="grey", linestyle=":", linewidth=1)
    ax_n.axvspan(0.5, 1, color=shade_colour, zorder=0)
    ax_n.text(0.75, 0.5, "Mismatch", transform=ax_n.transAxes,
              ha="center", va="center", fontsize=11, color="grey")
    ax_n.set(title="Normal SQuAD", xlabel="Parametric knowledge probability", ylabel="Density")
    # Draw grid *after* fills so it stays visible
    ax_n.grid(True, color="0.3", linestyle=":", linewidth=0.6, alpha=0.6, zorder=4)

    # Plot Context‑switched -------------------------------------------------
    kde_line(
        ax_p,
        df_p.loc[df_p.answer_correct, "p_parametric"],
        f"Correct (n={df_p.answer_correct.sum():,})",
        colours["correct"],
    )
    kde_line(
        ax_p,
        df_p.loc[~df_p.answer_correct, "p_parametric"],
        f"Incorrect (n={(~df_p.answer_correct).sum():,})",
        colours["incorrect"],
    )
    ax_p.axvline(0.5, color="grey", linestyle=":", linewidth=1)
    ax_p.axvspan(0, 0.5, color=shade_colour, zorder=0)
    ax_p.text(0.25, 0.5, "Mismatch", transform=ax_p.transAxes,
              ha="center", va="center", fontsize=11, color="grey")
    ax_p.set(title="Context‑switched SQuAD", xlabel="Parametric knowledge probability")
    ax_p.grid(True, color="0.3", linestyle=":", linewidth=0.6, alpha=0.6, zorder=4)

    # Harmonise y‑limits ----------------------------------------------------
    y_max = max(ax_n.get_ylim()[1], ax_p.get_ylim()[1])
    for ax in (ax_n, ax_p):
        ax.set_ylim(0, y_max)

    # Legend (inside right panel) ------------------------------------------
    ax_p.legend(loc="upper right", frameon=False)

    # Title (minimal padding) ----------------------------------------------
    fig.suptitle(
        "Distribution of parametric knowledge probabilities for normal vs. context‑switched SQuAD",
        fontsize=14,
        y=0.97,
    )
    fig.tight_layout()

    # ── Save ---------------------------------------------------------------
    out_fp = (
        Path(params.output_path) / "plots" / params.target_name /
        f"prob_density_SQuAD_{model.replace('-', '').replace('.', '')}_{classifier}.png"
    )
    out_fp.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_fp, dpi=300)
    plt.close(fig)
    print("• Plot saved to", out_fp)


# ── CLI --------------------------------------------------------------------

def main() -> None:
    classifiers = ["logreg", "mlp"]
    models = ["Llama-3.1-8B", "Mistral-7B-v0.1"]
    for classifier in classifiers:
        for model in models:
            create_plot(model, classifier)


if __name__ == "__main__":
    main()
