import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

from attriwiki_probe.all_layer_linear import WeightedAggLogReg
from src.classification.data import undersample_dataset

_PROBE_PATHS = {
    "Mistral-7B-v0.1":   "attriwiki_probe/Mistral-7B-v0.1/weighted_agg_logreg_first_token_generation.joblib",
    "Meta-Llama-3.1-8B": "attriwiki_probe/Llama-3.1-8B/weighted_agg_logreg_first_token_generation.joblib",
    "Llama-3.1-8B":      "attriwiki_probe/Llama-3.1-8B/weighted_agg_logreg_first_token_generation.joblib",
    "Qwen2.5-7B":        "attriwiki_probe/Qwen2.5-7B/weighted_agg_logreg_first_token_generation.joblib",
}
_LABEL_ENCODER = {"CK": 1, "PK": 0}


def _load_probe(probe_path: str, device: str) -> WeightedAggLogReg:
    payload = joblib.load(probe_path)
    probe = WeightedAggLogReg(num_layers=payload["num_layers"], hidden_dim=payload["hidden_dim"])
    probe.load_state_dict(payload["state_dict"])
    probe.to(device).eval()
    return probe


def _conf_histogram(scores: np.ndarray, bins: int = 5) -> str:
    """Compact unicode bar chart of confidence scores in [0, 1]."""
    counts, edges = np.histogram(scores, bins=bins, range=(0.0, 1.0))
    chars = "▁▂▃▄▅▆▇█"
    max_c = counts.max() or 1
    bars = "".join(chars[int(c / max_c * 7)] for c in counts)
    labels = "  ".join(f"{edges[i]:.1f}-{edges[i+1]:.1f}:{counts[i]}" for i in range(len(counts)))
    return f"{bars}  [{labels}]"


def _eval_group(
    probe: WeightedAggLogReg,
    test_df: pd.DataFrame,
    relation_group_id: str,
    device: str,
) -> dict:
    if len(test_df) == 0:
        return {"relation_group_id": relation_group_id, "accuracy": None, "macro_f1": None, "nb_test_examples": 0}

    X = torch.stack(test_df["generation_hidden_states"].tolist()).float().to(device)
    y = np.array([_LABEL_ENCODER[ks] for ks in test_df["knowledge_source"]])

    with torch.no_grad():
        logits = probe(X)
        confidences = torch.sigmoid(logits).cpu().numpy()   # P(CK) for each example

        # Weighted aggregated hidden state (before dropout/classifier) — used for norm analysis.
        # This is what the linear head actually operates on.
        norm_h = probe._norm(X)
        weights = torch.softmax(probe.layer_logits, dim=0)
        agg = torch.einsum("blh,l->bh", norm_h, weights).cpu().numpy()  # (n, hidden_dim)

    preds = (confidences > 0.5).astype(int)
    acc = accuracy_score(y, preds)
    _, _, f1_arr, _ = precision_recall_fscore_support(y, preds, labels=[0, 1], zero_division=0)

    # rows = true class (0=PK, 1=CK), cols = predicted class
    cm = confusion_matrix(y, preds, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    ck_mask = y == 1
    pk_mask = y == 0
    agg_norms = np.linalg.norm(agg, axis=1)

    # Annotate a local copy for per-relation and error analysis
    annotated = test_df.reset_index(drop=True).copy()
    annotated["_y"] = y
    annotated["_pred"] = preds
    annotated["_conf"] = confidences
    annotated["_correct"] = preds == y

    # Per-relation stats — reuses already-computed preds/confs, no extra forward pass needed
    per_relation = {}
    if "rel_lemma" in annotated.columns:
        for rel, sub in annotated.groupby("rel_lemma"):
            sub_y = sub["_y"].values
            sub_p = sub["_pred"].values
            per_relation[rel] = {
                "n": len(sub),
                "n_pk_true": int((sub_y == 0).sum()),
                "n_ck_true": int((sub_y == 1).sum()),
                "acc": float(accuracy_score(sub_y, sub_p)),
                "pct_pred_ck": float(sub_p.mean()),
                "mean_conf": float(sub["_conf"].mean()),
            }

    # Up to 8 misclassified examples sorted by confidence (most confident mistakes first)
    wrong = annotated[~annotated["_correct"]].sort_values("_conf", ascending=False)
    misclassified = []
    for _, row in wrong.head(8).iterrows():
        entry = {
            "true": "CK" if row["_y"] == 1 else "PK",
            "pred": "CK" if row["_pred"] == 1 else "PK",
            "conf": float(row["_conf"]),
        }
        for col in ("statement_subject", "rel_lemma", "parametric_knowledge_object",
                    "counter_knowledge_object", "output_object"):
            if col in row.index:
                entry[col] = row[col]
        misclassified.append(entry)

    return {
        # Core metrics (also written to CSV)
        "relation_group_id": relation_group_id,
        "accuracy": float(acc),
        "macro_f1": float(f1_arr.mean()),
        "nb_test_examples": len(test_df),
        # Confusion matrix counts
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        # Per-class recall
        "recall_pk": float(tn / (tn + fp)) if (tn + fp) > 0 else None,
        "recall_ck": float(tp / (tp + fn)) if (tp + fn) > 0 else None,
        # Prediction bias
        "pct_predicted_ck": float(preds.mean()),
        # Confidence distributions — mean ± std of P(CK) split by true class
        "mean_conf_ck": float(confidences[ck_mask].mean()) if ck_mask.any() else None,
        "std_conf_ck":  float(confidences[ck_mask].std())  if ck_mask.any() else None,
        "mean_conf_pk": float(confidences[pk_mask].mean()) if pk_mask.any() else None,
        "std_conf_pk":  float(confidences[pk_mask].std())  if pk_mask.any() else None,
        # Compact histograms for printing
        "conf_hist_ck": _conf_histogram(confidences[ck_mask]) if ck_mask.any() else None,
        "conf_hist_pk": _conf_histogram(confidences[pk_mask]) if pk_mask.any() else None,
        # Aggregated hidden-state norm — if CK and PK norms are identical the linear head has no signal
        "mean_agg_norm":    float(agg_norms.mean()),
        "mean_agg_norm_ck": float(agg_norms[ck_mask].mean()) if ck_mask.any() else None,
        "mean_agg_norm_pk": float(agg_norms[pk_mask].mean()) if pk_mask.any() else None,
        # Nested breakdowns (not written to CSV)
        "per_relation": per_relation,
        "misclassified_examples": misclassified,
    }


def _print_probe_stats(probe_path: str) -> None:
    """Print learned probe parameters: layer weights, classifier bias, weight norm."""
    payload = joblib.load(probe_path)
    probe = WeightedAggLogReg(num_layers=payload["num_layers"], hidden_dim=payload["hidden_dim"])
    probe.load_state_dict(payload["state_dict"])
    probe.eval()

    layer_weights = torch.softmax(probe.layer_logits.detach(), dim=0).numpy()
    top_idx = np.argsort(layer_weights)[::-1][:5]
    bias = probe.classifier.bias.item()
    w_norm = probe.classifier.weight.norm().item()

    chars = "▁▂▃▄▅▆▇█"
    w_min, w_max = layer_weights.min(), layer_weights.max()
    bar = "".join(chars[int(((x - w_min) / (w_max - w_min + 1e-12)) * 7)] for x in layer_weights)

    print("\n" + "=" * 60)
    print("PROBE PARAMETERS")
    print("=" * 60)
    print(f"  num_layers={payload['num_layers']}  hidden_dim={payload['hidden_dim']}  token_key={payload.get('token_key','?')}")
    print(f"  Classifier bias : {bias:+.4f}  {'(→ leans PK at neutral input)' if bias < 0 else '(→ leans CK at neutral input)'}")
    print(f"  Classifier |w|  : {w_norm:.4f}")
    print(f"  Layer weights   : {bar}")
    print(f"  Top-5 layers    : " + "  ".join(f"L{i}({layer_weights[i]:.3f})" for i in top_idx))
    print("=" * 60)


def _print_group_debug(result: dict) -> None:
    """Print full debug statistics for a single relation group."""
    grp = result["relation_group_id"]
    n = result["nb_test_examples"]
    if n == 0:
        print(f"\n  {grp}: 0 examples — skipped")
        return

    print(f"\n  ┌─ {grp}  (n={n}) {'─'*(50 - len(grp))}")
    print(f"  │  Accuracy {result['accuracy']:.3f}   Macro-F1 {result['macro_f1']:.3f}")

    tn, fp, fn, tp = result["tn"], result["fp"], result["fn"], result["tp"]
    r_pk = result["recall_pk"]
    r_ck = result["recall_ck"]
    pct_ck = result["pct_predicted_ck"]
    print(f"  │")
    print(f"  │  Confusion matrix          pred PK   pred CK")
    print(f"  │    true PK  ({tn+fp:3d} total) :  {tn:5d}     {fp:5d}   recall={r_pk:.3f}")
    print(f"  │    true CK  ({fn+tp:3d} total) :  {fn:5d}     {tp:5d}   recall={r_ck:.3f}")
    print(f"  │  Prediction split: {pct_ck*100:.1f}% CK  /  {(1-pct_ck)*100:.1f}% PK")

    print(f"  │")
    print(f"  │  Confidence P(CK) by true class:")
    if result["mean_conf_ck"] is not None:
        print(f"  │    CK examples: mean={result['mean_conf_ck']:.3f} ± {result['std_conf_ck']:.3f}   {result['conf_hist_ck']}")
    if result["mean_conf_pk"] is not None:
        print(f"  │    PK examples: mean={result['mean_conf_pk']:.3f} ± {result['std_conf_pk']:.3f}   {result['conf_hist_pk']}")
    print(f"  │    (separation = {abs(result['mean_conf_ck'] - result['mean_conf_pk']):.3f}  — higher is better)")

    print(f"  │")
    mn = result["mean_agg_norm"]
    mn_ck = result["mean_agg_norm_ck"]
    mn_pk = result["mean_agg_norm_pk"]
    print(f"  │  Aggregated hidden-state norm: mean={mn:.3f}  CK={mn_ck:.3f}  PK={mn_pk:.3f}")
    print(f"  │    (large norm difference suggests CK/PK live in different directions)")

    if result.get("per_relation"):
        print(f"  │")
        print(f"  │  Per-relation (within this test set):")
        print(f"  │    {'relation':<35s}  n   PK  CK   acc   %predCK  meanConf")
        for rel, s in sorted(result["per_relation"].items(), key=lambda x: -x[1]["n"]):
            print(f"  │    {rel:<35s}  {s['n']:3d}  {s['n_pk_true']:2d}  {s['n_ck_true']:2d}  "
                  f"{s['acc']:.3f}   {s['pct_pred_ck']:.2f}     {s['mean_conf']:.3f}")

    if result.get("misclassified_examples"):
        print(f"  │")
        print(f"  │  Most confident mistakes (true→pred, conf):")
        for ex in result["misclassified_examples"]:
            subj = ex.get("statement_subject", "?")
            rel  = ex.get("rel_lemma", "?")
            pk   = ex.get("parametric_knowledge_object", "?")
            ck   = ex.get("counter_knowledge_object", "?")
            out  = ex.get("output_object", "?")
            print(f"  │    [{ex['true']}→{ex['pred']} conf={ex['conf']:.3f}]  {subj!r}  {rel}"
                  f"  PK={pk!r}  CK={ck!r}  out={out!r}")

    print(f"  └{'─'*60}")


def perform_weighted_agg_logreg_classification(
    prompting_results: pd.DataFrame,
    model_name: str,
    device: str = "cpu",
) -> list[dict]:
    """Zero-shot evaluation of the pre-trained WeightedAggLogReg on generation-time hidden states.

    Uses the same per-relation-group test sets as perform_classification_by_relation_group so
    results are directly comparable against Tighidet's per-layer logistic regression.
    """
    if model_name not in _PROBE_PATHS:
        raise ValueError(f"No ATTRIWIKI probe registered for model '{model_name}'. Available: {list(_PROBE_PATHS)}")
    probe_path = _PROBE_PATHS[model_name]
    probe = _load_probe(probe_path, device)
    prompting_results = prompting_results.reset_index(drop=True)
    metrics = []

    for relation_group_id in prompting_results.relation_group_id.unique():
        test_df = undersample_dataset(
            prompting_results[prompting_results.relation_group_id == relation_group_id]
        )
        metrics.append(_eval_group(probe, test_df, relation_group_id, device))

    return metrics


def print_weighted_agg_summary(metrics: list[dict], model_name: str) -> None:
    _print_probe_stats(_PROBE_PATHS[model_name])

    valid = [m for m in metrics if m.get("accuracy") is not None]
    weights = np.array([m["nb_test_examples"] for m in valid], dtype=float)
    mean_acc = np.average([m["accuracy"] for m in valid], weights=weights)
    mean_f1  = np.average([m["macro_f1"]  for m in valid], weights=weights)

    print("\n" + "=" * 60)
    print("WEIGHTED AGG LOGREG SUMMARY (pre-trained probe)")
    print("=" * 60)
    print(f"  Weighted mean accuracy : {mean_acc:.3f}")
    print(f"  Weighted mean macro-F1 : {mean_f1:.3f}")
    print()
    print(f"  {'relation group':<40s}  acc    f1     recall_PK  recall_CK  %predCK")
    for m in sorted(valid, key=lambda x: -x["accuracy"]):
        print(f"  {m['relation_group_id']:<40s}  {m['accuracy']:.3f}  {m['macro_f1']:.3f}  "
              f"{m['recall_pk']:.3f}      {m['recall_ck']:.3f}      {m['pct_predicted_ck']*100:.0f}%")
    print("=" * 60)

    print("\nDETAILED DEBUG STATS PER GROUP:")
    for m in metrics:
        _print_group_debug(m)
    print()


def save_weighted_agg_metrics(
    metrics: list[dict],
    model_name: str,
    permanent_path: str,
    test_on_head: bool,
    n_head: int,
) -> None:
    # Only write scalar columns to CSV — nested per_relation / misclassified_examples are print-only
    scalar_keys = [
        "relation_group_id", "accuracy", "macro_f1", "nb_test_examples",
        "tn", "fp", "fn", "tp", "recall_pk", "recall_ck", "pct_predicted_ck",
        "mean_conf_ck", "std_conf_ck", "mean_conf_pk", "std_conf_pk",
        "mean_agg_norm", "mean_agg_norm_ck", "mean_agg_norm_pk",
    ]
    rows = [{k: m.get(k) for k in scalar_keys} for m in metrics]

    save_path = f"{permanent_path}/classification_metrics/{model_name}_weighted_agg_logreg_metrics"
    if test_on_head:
        save_path += f"_n_head={n_head}"
    save_path += ".csv"
    pd.DataFrame(rows).to_csv(save_path, index=False)
    print(f"WeightedAggLogReg metrics saved to {save_path}")
