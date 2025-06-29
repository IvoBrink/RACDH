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
    Settings,
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
        default=Settings.MODEL,
        help="Classifier model name (joblib in models/)",
    )
    p.add_argument(
        "--plot-dir",
        default=Settings.PLOTS_DIR,
        type=Path,
        help="Directory to write plots/results to",
    )
    return p.parse_args()


# --------------------------------------------------------------------------- #
#  Helpers                                                                     #
# --------------------------------------------------------------------------- #

def _load_json(path: Path) -> list[dict[str, Any]]:
    try:
        data = json.loads(path.read_text())
    except Exception as e:
        sys.exit(f"✗  Failed to read {path}: {e}")
    if not isinstance(data, list):
        sys.exit(f"✗  {path} does not contain a list of JSON objects")
    return data


def _infer_dataset(path: Path, records_sample: list[dict[str, Any]]) -> Dataset:
    """Infer dataset type from folder/filename clues *or* record structure."""
    parts_lower = {p.lower() for p in path.parts}
    if any(k in parts_lower for k in ("webq", "webquestions")):
        return "webq"
    if "squad" in parts_lower:
        return "squad"
    # Fallback heuristic: SQuAD records carry a context string
    has_context = any(r.get("context") for r in records_sample)
    return "squad" if has_context else "webq"


_SETTINGS_TOKEN_RE = re.compile(r"(?P<key>[A-Za-z0-9\-]+):(?P<val>[^_]+)")


def _coerce(v: str):
    """Best‑effort string→bool/int/str coercion."""
    lv = v.lower()
    if lv in {"true", "false"}:
        return lv == "true"
    if v.isdigit():
        return int(v)
    return v



def _prepare_dataframe(records: list[dict[str, Any]]):
    df = pd.DataFrame(records)
    df["hidden_np"] = df["hidden"].apply(lambda v: np.asarray(v, dtype=np.float32))
    return df


# --------------------------------------------------------------------------- #
#  Main                                                                        #
# --------------------------------------------------------------------------- #

def main() -> None:
    a = _args()

    # --------------------------------------------------------------- #
    #  Classifier setup                                              #
    # --------------------------------------------------------------- #

    TAG_SUFFIX = {
        "all_context_switched": "_parametric",
        "only_prior_no_switch": "_prior_no_switch",
        "only_prior_entity_decoy": "_prior_entity_decoy",
    }


    data_path = f"RACDH/data/{a.dataset}/{params.target_name}/infer_{a.token_key}_balanced_2000"
    for flag, suffix in TAG_SUFFIX.items():
        if getattr(a, flag):
            data_path += suffix
            break
    data_path += ".json"


    model_path = Path(params.output_path) / "models" / params.target_name / f"{a.model}.joblib"
    clf_cls = StackHiddenStateClassifier if "agg" in a.model else HiddenStateClassifier
    clf = clf_cls(model_path)

    # --------------------------------------------------------------- #
    #  Gather & bucket records                                        #
    # --------------------------------------------------------------- #
    grouped: dict[Dataset, list[dict[str, Any]]] = {"squad": [], "webq": []}


    

    for file_ in a.inputs:
        p = Path(file_)
        if not p.exists():
            sys.exit(f"✗  File not found: {p}")

        file_settings = _parse_settings_from_filename(p)
        recs = _load_json(p)
        # attach file‑level settings to every record so they survive into DF
        for r in recs:
            r.update(file_settings)
        ds = _infer_dataset(p, recs[:50])
        grouped[ds].extend(recs)
        print(f"✓  Loaded {len(recs):,} {ds.upper()} records from {p.name}")

    if not any(grouped.values()):
        sys.exit("✗  No records to evaluate.")

    # --------------------------------------------------------------- #
    #  Evaluate per dataset                                          #
    # --------------------------------------------------------------- #
    apply_kpmg_theme()
    colours = {
        "correct": "#66d575",   # teal
        "incorrect": "#f24822", # orange
        "prior": "#1c47e3",    # blue
    }

    a.plot_dir.mkdir(parents=True, exist_ok=True)

    for ds_name, records in grouped.items():
        if not records:
            continue

        print(f"\n===== DATASET: {ds_name.upper()} ================================")
        df = _prepare_dataframe(records)

        # Predictions – convert NumPy array → torch.Tensor because the
        # classifier expects a tensor with an `.unsqueeze()` method
        preds = df["hidden_np"].apply(lambda v: clf.predict(torch.from_numpy(v)))
        pred_df = pd.DataFrame(list(preds))
        
        df = pd.concat([df, pred_df], axis=1)

        # Ground‑truth label logic
        correct_lbl, incorrect_lbl = (
            ("Parametric", "Contextual") if ds_name == "webq" or file_settings["context"] == "all-switched" else ("Contextual", "Parametric")
        )

        # Reports
        print("\n— Correct answers (expected →", correct_lbl, ") —")
        print(f"Correct answers length: {len(df[df.answer_correct])}")
        print(
            classification_report(
                [correct_lbl] * len(df[df.answer_correct]),
                df[df.answer_correct]["label"],
                labels=["Parametric", "Contextual"],
                digits=3
            )
        )

        print("\n— Incorrect answers (expected →", incorrect_lbl, ") —")
        print(f"Incorrect answers length: {len(df[~df.answer_correct])}")
        print(
            classification_report(
                [incorrect_lbl] * len(df[~df.answer_correct]),
                df[~df.answer_correct]["label"],
                labels=["Parametric", "Contextual"],
                digits=3
            )
        )

 
        if file_settings["context"] == "all-switched":
            identifier_name = f"{ds_name}_{Settings.MODEL}_parametric_testing"
        elif file_settings["context"] == "prior-switch":
            identifier_name = f"{ds_name}_{Settings.MODEL}_prior_switch"
            prior_title = " with random context"
        elif file_settings["context"] == "prior-entity-bait":
            identifier_name = f"{ds_name}_{Settings.MODEL}_prior_entity_bait"
            prior_title = " with random context and decoy entity"
        elif file_settings["context"] == "normal":
            identifier_name = f"{ds_name}_{Settings.MODEL}_normal"
            prior_title = " with relevant context"
        else:
            raise RuntimeError("Unknown context value in file_settings['context']: {}".format(file_settings["context"]))

        # KDE plot
        fig, ax = plt.subplots(figsize=(8, 4.5))
        kde_line(ax, df[df.answer_correct]["p_parametric"], "Answered correctly", colours["correct"])
        kde_line(ax, df[~df.answer_correct]["p_parametric"], "Answered incorrectly", colours["incorrect"])
        if df["prior_knowledge"].any():
            kde_line(ax, df[df.prior_knowledge]["p_parametric"], "Marked as prior knowledge" + prior_title, colours["prior"])
            print(f"Prior knowledge length: {len(df[df.prior_knowledge])}")
            print("\n— Prior knowledge (expected →", "Parametric", ") —")
            print(
            classification_report(
                ["Parametric"] * len(df[df.prior_knowledge]),
                df[df.prior_knowledge]["label"],
                labels=["Parametric", "Contextual"],
                digits=3
            )
        )

        ax.axvline(Settings.PARAM_THRESHOLD, color="grey", linestyle=":", linewidth=1)
        ax.set(
            xlabel="Parametric knowledge probability",
            ylabel="Density",
            title=f"{ds_name.upper()} – Distribution of parametric knowledge probability",
        )
        ax.legend()
        fig.tight_layout()

        plot_path = a.plot_dir / params.target_name / f"parametric_probability_density_{identifier_name}.png"
        fig.savefig(plot_path, dpi=300)
        plt.close(fig)
        print("• Plot saved to", plot_path)

        
        results_path = a.plot_dir / params.target_name / f"results_{identifier_name}.json"
        df.drop(columns=["hidden_np", "hidden"]).to_json(results_path, orient="records", indent=2)
        print("• Results JSON saved to", results_path)

    print("\n✓  Evaluation complete.")


if __name__ == "__main__":
    main()
