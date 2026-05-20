"""Per-relation-group breakdown table for the appendix.

For each model, produces a LaTeX table with bootstrap macro-F1 ± std for
both the Tighidet (best-layer) probe and the ATTRIWIKI probe, broken down
by relation group.

Usage:
    PYTHONPATH=. python scripts/per_group_table.py
    PYTHONPATH=. python scripts/per_group_table.py --model_name Mistral-7B-v0.1
    PYTHONPATH=. python scripts/per_group_table.py --all_models
"""

import argparse

import numpy as np
import pandas as pd


MODELS = ["Mistral-7B-v0.1", "Llama-3.1-8B", "Qwen2.5-7B"]

GROUP_DISPLAY = {
    "geographic-geopolitic-language": "Geographic",
    "corporate-products-employment": "Corporate",
    "media": "Media",
    "religion": "Religion",
    "hierarchy": "Hierarchy",
    "occupy-position": "Occupy-position",
    "play-instrument": "Play-instrument",
    "naming-reference": "Naming-reference",
}


def _macro_f1_from_cm(tp, tn, fp, fn):
    f1_ck = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0
    f1_pk = 2 * tn / (2 * tn + fn + fp) if (2 * tn + fn + fp) > 0 else 0.0
    return (f1_ck + f1_pk) / 2


def _bootstrap_group(n, tp, tn, fp, fn, n_bootstrap, rng):
    probs = np.array([tp, tn, fp, fn], dtype=float) / n
    f1s = [
        _macro_f1_from_cm(*rng.multinomial(n, probs))
        for _ in range(n_bootstrap)
    ]
    return np.array(f1s)


def _load_tighidet(csv_path):
    df = pd.read_csv(csv_path)
    if "accuracy" not in df.columns and "success_rate" in df.columns:
        df = df.rename(columns={"success_rate": "accuracy"})
    return df


def _best_layer(df):
    has_cm = {"tp", "tn", "fp", "fn"}.issubset(df.columns) and df["tp"].notna().any()
    valid = df.dropna(subset=["accuracy"])
    if has_cm:
        by_layer = valid.groupby("layer").apply(
            lambda g: np.average(g["macro_f1"], weights=g["nb_test_examples"])
        )
    else:
        by_layer = valid.groupby("layer").apply(
            lambda g: np.average(g["accuracy"], weights=g["nb_test_examples"])
        )
    return int(by_layer.idxmax())


def per_group_stats(model_name, tighidet_csv, attriwiki_csv, n_bootstrap=1000, seed=42):
    rng = np.random.default_rng(seed)

    t_df = _load_tighidet(tighidet_csv)
    best_layer = _best_layer(t_df.dropna(subset=["accuracy"]))
    t_best = t_df[t_df["layer"] == best_layer]

    a_df = pd.read_csv(attriwiki_csv)

    all_groups = sorted(
        set(t_df["relation_group_id"].dropna()) | set(a_df["relation_group_id"].dropna())
    )

    rows = []
    for group in all_groups:
        t_row = t_best[t_best["relation_group_id"] == group]
        a_row = a_df[a_df["relation_group_id"] == group]

        # Tighidet
        if not t_row.empty and pd.notna(t_row.iloc[0]["accuracy"]) and int(t_row.iloc[0]["nb_test_examples"]) > 0:
            r = t_row.iloc[0]
            n_t = int(r["nb_test_examples"])
            tp, tn = int(r["tp"]), int(r["tn"])
            fp, fn = int(r["fp"]), int(r["fn"])
            t_f1s = _bootstrap_group(n_t, tp, tn, fp, fn, n_bootstrap, rng)
            t_mean, t_std = np.mean(t_f1s), np.std(t_f1s)
        else:
            n_t, t_mean, t_std = 0, None, None

        # ATTRIWIKI
        if not a_row.empty and pd.notna(a_row.iloc[0].get("accuracy")) and int(a_row.iloc[0]["nb_test_examples"]) > 0:
            r = a_row.iloc[0]
            n_a = int(r["nb_test_examples"])
            tp, tn = int(r["tp"]), int(r["tn"])
            fp, fn = int(r["fp"]), int(r["fn"])
            a_f1s = _bootstrap_group(n_a, tp, tn, fp, fn, n_bootstrap, rng)
            a_mean, a_std = np.mean(a_f1s), np.std(a_f1s)
        else:
            n_a, a_mean, a_std = 0, None, None

        rows.append({
            "group": group,
            "n_tighidet": n_t,
            "t_f1_mean": t_mean,
            "t_f1_std": t_std,
            "n_attriwiki": n_a,
            "a_f1_mean": a_mean,
            "a_f1_std": a_std,
        })

    return rows, best_layer


def fmt_cell(mean, std):
    if mean is None:
        return r"\multicolumn{1}{c}{---}"
    return rf"{mean:.3f} $\pm$ {std:.3f}"


def print_latex_table(model_name, rows, best_layer):
    short = model_name.replace("-v0.1", "").replace("-", " ").replace(".", ".")
    print(r"\begin{table}[h]")
    print(r"  \centering")
    print(r"  \small")
    print(r"  \begin{tabular}{lrrr}")
    print(r"    \toprule")
    print(
        r"    \textbf{Relation group}"
        r" & $n$ "
        rf"& \textbf{{Tighidet (L{best_layer})}}"
        r" & \textbf{ATTRIWIKI} \\"
    )
    print(r"    \midrule")

    rows = [r for r in rows if r["n_tighidet"] > 0 or r["n_attriwiki"] > 0]

    for r in rows:
        label = GROUP_DISPLAY.get(r["group"], r["group"])
        # use Tighidet n as the sample size (same dataset); fall back to ATTRIWIKI
        n = r["n_tighidet"] if r["n_tighidet"] > 0 else r["n_attriwiki"]
        t_cell = fmt_cell(r["t_f1_mean"], r["t_f1_std"])
        a_cell = fmt_cell(r["a_f1_mean"], r["a_f1_std"])
        print(rf"    {label} & {n} & {t_cell} & {a_cell} \\")

    print(r"    \bottomrule")
    print(r"  \end{tabular}")
    print(
        rf"  \caption{{Per-group macro-F1 (mean $\pm$ bootstrap std, $B=1000$) for \textbf{{{short}}}."
        rf" Tighidet uses the best overall layer (L{best_layer}).}}"
    )
    print(rf"  \label{{tab:per-group-{model_name.lower().replace('.', '').replace('-', '-')}}}")
    print(r"\end{table}")
    print()


def run_model(model_name, metrics_dir, token_position, module, n_bootstrap):
    tighidet_csv = (
        f"{metrics_dir}/{model_name}_classification_metrics_by_relation_groups"
        f"_{token_position}_{module}.csv"
    )
    attriwiki_csv = f"{metrics_dir}/{model_name}_weighted_agg_logreg_metrics.csv"

    rows, best_layer = per_group_stats(
        model_name, tighidet_csv, attriwiki_csv, n_bootstrap=n_bootstrap
    )
    print_latex_table(model_name, rows, best_layer)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default=None)
    parser.add_argument("--all_models", action="store_true")
    parser.add_argument("--token_position", default="relation_query")
    parser.add_argument("--module", default="mlps")
    parser.add_argument("--n_bootstrap", type=int, default=1000)
    parser.add_argument("--metrics_dir", default="classification_metrics")
    args = parser.parse_args()

    models = MODELS if args.all_models or args.model_name is None else [args.model_name]
    for m in models:
        run_model(m, args.metrics_dir, args.token_position, args.module, args.n_bootstrap)
