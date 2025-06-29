#!/usr/bin/env python
"""
Title-aware train/test split and simple Logistic Regression on the **last**
transformer layer representation (no layer weighting).  
Class imbalance handled via `class_weight='balanced'`.
"""

import os
import sys
import argparse
import joblib
import torch
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# ------------------------------------------------------------------ #
#  project-specific imports
# ------------------------------------------------------------------ #
sys.path.append(os.path.abspath("/home/ibrink/RACDH/RACDH/"))
from RACDH.config import params
from RACDH.data_generation.utils.reading_data import load_json
from RACDH.classification.utils import (
    groupwise_train_test_split,
    load_vectors_and_labels,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--token_key",
        choices=[
            "first_token_entity",
            "last_token_entity",
            "first_token_generation",
            "last_token_before_entity",
        ],
        default="first_token_generation",
        help="Which hidden-state stack to use",
    )
    parser.add_argument("--max_iter", type=int, default=1000)
    args = parser.parse_args()

    # -------------------------------------------------------------- #
    #  Load hidden states (+ metadata)
    # -------------------------------------------------------------- #
    hidden_states = torch.load(
        os.path.join(
            params.output_path,
            f"{params.target_name}/{params.instruct_name}/hiddens_all_2.pt",
        ),
        map_location="cpu",
    )
    meta = load_json(
        f"{params.target_name}/{params.instruct_name}/hiddens_metadata_all_2.json"
    )

    X, y, _, _ = load_vectors_and_labels(
        hidden_states, meta, args.token_key, reduce="stack"
    )  # X shape: (N, L, H)

    # ------ keep ONLY the last layer --------------------------------
    X_last = X[:, -1, :].numpy()        # (N, H)
    y      = y.numpy()

    print(f"Loaded {X_last.shape[0]} samples — last-layer dim = {X_last.shape[1]}")

    # -------------------------------------------------------------- #
    #  Title-aware train/test split
    # -------------------------------------------------------------- #
    train_idx, test_idx = groupwise_train_test_split(meta)
    X_train, X_test = X_last[train_idx], X_last[test_idx]
    y_train, y_test = y[train_idx],      y[test_idx]

    # -------------------------------------------------------------- #
    #  Logistic Regression (balanced)
    # -------------------------------------------------------------- #
    clf = Pipeline(
        steps=[
            ("scale", StandardScaler()),
            (
                "logreg",
                LogisticRegression(
                    max_iter=args.max_iter,
                    class_weight="balanced",
                    solver="lbfgs",
                ),
            ),
        ]
    )
    clf.fit(X_train, y_train)

    # -------------------------------------------------------------- #
    #  Evaluation
    # -------------------------------------------------------------- #
    y_pred = clf.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, labels=[0, 1], zero_division=0
    )

    print(f"\nTest accuracy: {acc:.3f}")

    # --- per-class scores -------------------------------------------------------
    print("\nClass-wise metrics on test set")
    print("  Class  Precision  Recall  F1")
    print(f"    0    {prec[0]:.3f}     {rec[0]:.3f}  {f1[0]:.3f}   (parametric)")
    print(f"    1    {prec[1]:.3f}     {rec[1]:.3f}  {f1[1]:.3f}   (contextual)")

    # --- macro F1 ---------------------------------------------------------------
    macro_f1 = (f1[0] + f1[1]) / 2        # arithmetic mean
    print(f"\nMacro-averaged F1: {macro_f1:.3f}")

    cm = confusion_matrix(y_test, y_pred)
    print(cm)

    # -------------------------------------------------------------- #
    #  Save the trained pipeline
    # -------------------------------------------------------------- #
    model_dir = f"RACDH/data/models/{params.target_name}"
    if not os.path.exists(model_dir):
        os.makedirs(model_dir, exist_ok=True)
    model_path = f"{model_dir}/logreg_last_layer_{args.token_key}.joblib"
    joblib.dump(clf, model_path)
    print(f"\nSaved pipeline to {model_path}")


if __name__ == "__main__":
    main()
