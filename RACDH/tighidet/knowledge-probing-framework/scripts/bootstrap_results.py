"""Bootstrap variance estimation for a single model.

Outputs one table: macro-F1 ± std for Tighidet (best layer) and ATTRIWIKI probe.
Groups are combined into a single pooled estimate weighted by sample size.

Usage:
    PYTHONPATH=. python scripts/bootstrap_results.py --model_name Mistral-7B-v0.1
    PYTHONPATH=. python scripts/bootstrap_results.py --model_name Qwen2.5-7B
"""

import argparse

import numpy as np
import pandas as pd


def _macro_f1_from_cm(tp: int, tn: int, fp: int, fn: int) -> float:
    f1_ck = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0
    f1_pk = 2 * tn / (2 * tn + fn + fp) if (2 * tn + fn + fp) > 0 else 0.0
    return (f1_ck + f1_pk) / 2


def _pool_cm(df: pd.DataFrame) -> tuple[int, int, int, int, int]:
    """Sum confusion matrix counts across all groups. Returns (n, tp, tn, fp, fn)."""
    n  = int(df["nb_test_examples"].sum())
    tp = int(df["tp"].sum())
    tn = int(df["tn"].sum())
    fp = int(df["fp"].sum())
    fn = int(df["fn"].sum())
    return n, tp, tn, fp, fn


def _bootstrap_cm(n: int, tp: int, tn: int, fp: int, fn: int,
                  n_bootstrap: int, rng: np.random.Generator) -> np.ndarray:
    """Multinomial bootstrap of macro-F1 from pooled confusion matrix."""
    probs = np.array([tp, tn, fp, fn], dtype=float) / n
    f1s = []
    for _ in range(n_bootstrap):
        tp_b, tn_b, fp_b, fn_b = rng.multinomial(n, probs)
        f1s.append(_macro_f1_from_cm(tp_b, tn_b, fp_b, fn_b))
    return np.array(f1s)


def _load_tighidet(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if "accuracy" not in df.columns and "success_rate" in df.columns:
        df = df.rename(columns={"success_rate": "accuracy"})
    return df.dropna(subset=["accuracy"])


def bootstrap_model(
    model_name: str,
    tighidet_csv: str,
    attriwiki_csv: str,
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> None:
    rng = np.random.default_rng(seed)

    # ── Tighidet: pick best layer by weighted macro-F1 ──────────────────────
    t_df = _load_tighidet(tighidet_csv)
    has_cm = {"tp", "tn", "fp", "fn"}.issubset(t_df.columns) and t_df["tp"].notna().any()

    if has_cm:
        by_layer_f1 = t_df.groupby("layer").apply(
            lambda g: np.average(g["macro_f1"], weights=g["nb_test_examples"])
        )
    else:
        by_layer_f1 = t_df.groupby("layer").apply(
            lambda g: np.average(g["accuracy"], weights=g["nb_test_examples"])
        )
    best_layer = int(by_layer_f1.idxmax())
    best_df = t_df[t_df["layer"] == best_layer]
    t_n = int(best_df["nb_test_examples"].sum())

    if has_cm:
        _, tp, tn, fp, fn = _pool_cm(best_df)
        t_f1s = _bootstrap_cm(t_n, tp, tn, fp, fn, n_bootstrap, rng)
        t_f1_note = f"L{best_layer}, n={t_n}"
    else:
        t_f1s = None
        t_f1_note = f"L{best_layer}, n={t_n}  [F1 bootstrap unavailable — re-run to save confusion matrix]"

    # ── ATTRIWIKI: pool all groups ───────────────────────────────────────────
    a_df = pd.read_csv(attriwiki_csv).dropna(subset=["accuracy"])
    a_n, tp, tn, fp, fn = _pool_cm(a_df)
    a_f1s = _bootstrap_cm(a_n, tp, tn, fp, fn, n_bootstrap, rng)

    # ── Print table ──────────────────────────────────────────────────────────
    print(f"\nModel: {model_name}  |  bootstrap n={n_bootstrap}")
    print("=" * 65)
    print(f"  {'probe':<20s}  {'n':>6s}  {'macro-F1':>10s}  {'95% CI':>20s}  note")
    print("-" * 65)

    if t_f1s is not None:
        print(f"  {'Tighidet':<20s}  {t_n:>6d}  "
              f"{np.mean(t_f1s):.3f} ± {np.std(t_f1s):.3f}  "
              f"[{np.percentile(t_f1s, 2.5):.3f}, {np.percentile(t_f1s, 97.5):.3f}]  "
              f"best layer L{best_layer}")
    else:
        print(f"  {'Tighidet':<20s}  {t_n:>6d}  {'n/a (no CM)':>10s}  {t_f1_note}")

    print(f"  {'ATTRIWIKI':<20s}  {a_n:>6d}  "
          f"{np.mean(a_f1s):.3f} ± {np.std(a_f1s):.3f}  "
          f"[{np.percentile(a_f1s, 2.5):.3f}, {np.percentile(a_f1s, 97.5):.3f}]")
    print("=" * 65)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--token_position", default="relation_query")
    parser.add_argument("--module", default="mlps")
    parser.add_argument("--n_bootstrap", type=int, default=1000)
    parser.add_argument("--metrics_dir", default="classification_metrics")
    args = parser.parse_args()

    tighidet_csv = (
        f"{args.metrics_dir}/{args.model_name}_classification_metrics_by_relation_groups"
        f"_{args.token_position}_{args.module}.csv"
    )
    attriwiki_csv = f"{args.metrics_dir}/{args.model_name}_weighted_agg_logreg_metrics.csv"

    bootstrap_model(
        model_name=args.model_name,
        tighidet_csv=tighidet_csv,
        attriwiki_csv=attriwiki_csv,
        n_bootstrap=args.n_bootstrap,
    )
