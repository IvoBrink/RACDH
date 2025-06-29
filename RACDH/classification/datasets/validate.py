# evaluate.py
#!/usr/bin/env python3
"""evaluate.py – *lightweight* post‑processing for `generate.py` outputs
-----------------------------------------------------------------------
Reads one or more `infer_*.json` files (from both **SQuAD** & **WebQuestions**)
and evaluates their hidden‑state vectors with a saved classifier.

**New in this version**
=======================
* **Filename‑encoded settings are parsed** (e.g. model name, `par-testing`,
  `entity-bait`, `sample`) and stored alongside each record, so they remain
  available throughout the script – in the pandas DataFrame, in printed
  reports, and in the final results JSON.
* Works with any mix of SQuAD & WebQ files – records get auto‑routed to the
  correct dataset bucket and evaluated separately.
* Same outputs as before → classification reports to stdout, per‑dataset KDE
  plots & consolidated results JSON files written to `--plot-dir`.

Filename conventions recognised
-------------------------------
```
SQuAD: infer_<MODEL>_par-testing:<bool>_entity-bait:<bool>_<SAMPLE>.json
WebQ : infer_<MODEL>_<SAMPLE>.json
```
Other patterns fall back gracefully; anything of the form `key:value` or a lone
integer token (`sample`) is captured.
"""
from __future__ import annotations

import argparse, json, sys, re, os
from pathlib import Path
from typing import Any, Literal

import joblib
import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report

from utils import (
    HiddenStateClassifier,
    StackHiddenStateClassifier,
    kde_line,
    apply_kpmg_theme,
)
sys.path.append(os.path.abspath("/home/ibrink/RACDH/RACDH/"))
from RACDH.config import params

Dataset = Literal["squad", "webq"]

# --------------------------------------------------------------------------- #
#  CLI helpers                                                                 #
# --------------------------------------------------------------------------- #

def _args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluate hidden‑state classifier on JSON inference files",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--dataset", default="squad", help="'squad', 'webq', or a JSON file")
    grp = p.add_mutually_exclusive_group()
    grp.add_argument("--all-context-switched", action="store_true")
    grp.add_argument("--only-prior-no-switch", action="store_true")
    grp.add_argument("--only-prior-entity-decoy", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--token-key",
        choices=[
            "first_token_entity",
            "last_token_entity",
            "first_token_generation",
            "last_token_before_entity",
        ],
        default="first_token_generation",
        help="Which hidden-state stack to use",
    )
    p.add_argument(
        "--model",
        help="Classifier model name (joblib in models/)",
    )
    return p.parse_args()



def _prepare_dataframe(records: list[dict[str, Any]]):
    df = pd.DataFrame(records)
    df["hidden_np"] = df["hidden"].apply(lambda v: np.asarray(v, dtype=np.float32))
    return df

# --------------------------------------------------------------------------- #
#  Main                                                                        #
# --------------------------------------------------------------------------- #

def main() -> None:
    a = _args()
    prior_mode = a.only_prior_no_switch or a.only_prior_entity_decoy

    # --------------------------------------------------------------- #
    #  Classifier setup                                              #
    # --------------------------------------------------------------- #

    TAG_SUFFIX = {
        "all_context_switched": "_parametric",
        "only_prior_no_switch": "_prior_no_switch",
        "only_prior_entity_decoy": "_prior_entity_decoy",
    }

    PLOT_SUFFIX = {
        "all_context_switched": "All contexts shuffled",
        "only_prior_no_switch": "All prior knowledge, original contexts",
        "only_prior_entity_decoy": "All prior knowledge, arbitrary context containing answer decoy",
    }

    plot_description = ""
    suffix_results = "_normal"
    model_abbrev = "mlp" if "mlp" in a.model else "logreg"
    data_path = f"RACDH/data/{a.dataset}/{params.target_name}/infer_{a.token_key}_balanced_2000"
    for flag, suffix in TAG_SUFFIX.items():
        if getattr(a, flag):
            plot_description = PLOT_SUFFIX[flag]
            suffix_results = suffix
            data_path += suffix
            break
    data_path += ".json"


    model_path = Path(params.output_path) / "models" / params.target_name / f"{a.model}.joblib"
    clf_cls = StackHiddenStateClassifier if "agg" in a.model else HiddenStateClassifier
    clf = clf_cls(model_path)
    print(f"• Loaded model {model_path}")


    # Load the data from the specified data_path
    with open(data_path, "r") as f:
        records = json.load(f)
    df = _prepare_dataframe(records)
    print(f"• Loaded records {data_path}")
    print(df)


    preds = df["hidden_np"].apply(lambda v: clf.predict(torch.from_numpy(v)))

    pred_df = pd.DataFrame(list(preds))
        
    df = pd.concat([df, pred_df], axis=1)

    df = df.drop(columns=["hidden", "hidden_np"])

    print(df)


    df_correct = df[df.answer_correct]

    df_incorrect = df[~df.answer_correct]

    for name, d in [("Correct", df_correct), ("Incorrect", df_incorrect)]:
        n_ctx = (d["label"] == "Contextual").sum()
        n_par = (d["label"] == "Parametric").sum()
        tot = n_ctx + n_par
        print(f"{name} ({len(d)}): \t {100*n_ctx/tot:.1f}% Contextual \t | {100*n_par/tot:.1f}% Parametric")


     # KDE plot
    apply_kpmg_theme()
    if prior_mode:
        colours = {
            "correct": "#00b8f9", 
            "incorrect": "#00318d", 
        }
    else:
        colours = {
            "correct": "#66d575",   # teal
            "incorrect": "#f24822", # orange
        }

    fig, ax = plt.subplots(figsize=(8, 4.5))
    n_correct = len(df[df.answer_correct])
    n_incorrect = len(df[~df.answer_correct])
    kde_line(
        ax,
        df[df.answer_correct]["p_parametric"],
        f"Answered correctly (n={n_correct:,})",
        colours["correct"]
    )
    kde_line(
        ax,
        df[~df.answer_correct]["p_parametric"],
        f"Answered incorrectly (n={n_incorrect:,})",
        colours["incorrect"]
    )

    ax.axvline(0.5, color="grey", linestyle=":", linewidth=1)
    ax.set(
        xlabel="Parametric knowledge probability",
        ylabel="Density",
        title=f"{a.dataset.upper()} – Distribution of parametric knowledge probability\n {plot_description}",
    )
    ax.legend()
    fig.tight_layout()

    plot_path = Path(params.output_path) / "plots" / params.target_name / f"probability_density_{a.dataset}_{model_abbrev}_{suffix_results}.png"
    fig.savefig(plot_path, dpi=300)
    plt.close(fig)
    print("• Plot saved to", plot_path)


    results_path = Path(params.output_path) / "plots" /  params.target_name / f"results_{a.dataset}_{model_abbrev}_{suffix_results}.json"
    df.to_json(results_path, orient="records", indent=2)
    print("• Results JSON saved to", results_path)

if __name__ == "__main__":
    main()
