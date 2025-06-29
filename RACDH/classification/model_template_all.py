from xgboost import XGBClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import accuracy_score, classification_report
import joblib
import numpy as np
import os
import sys
import torch
sys.path.append(os.path.abspath("/home/ibrink/RACDH/RACDH/"))
from RACDH.config import params                         # noqa: E402
from RACDH.data_generation.utils.reading_data import load_json  # noqa: E402
from RACDH.classification.utils import (
    groupwise_train_test_split,
    load_vectors_and_labels,
)

MODEL_DIR = os.path.join(params.output_path, "models")
os.makedirs(MODEL_DIR, exist_ok=True)


def main() -> None:
    # --------------------------------------------------------------
    #  Load hidden states & metadata
    # --------------------------------------------------------------
    hidden_states = torch.load(
        os.path.join(
            params.output_path,
            f"{params.target_name}/{params.instruct_name}/hiddens_all.pt",
        )
    )
    meta_hidden_states = load_json(
        f"{params.target_name}/{params.instruct_name}/hiddens_metadata_all.json"
    )

    # --------------------------------------------------------------
    #  Group-aware train/test split (by passage titles)
    # --------------------------------------------------------------
    train_idx, test_idx = groupwise_train_test_split(meta_hidden_states)

    # ---------- NEW KEYS THAT EXIST IN hiddens_rewrite.pt ----------
    TOKEN_KEYS = [
        "first_token_entity",
        "last_token_entity",
        "first_token_generation",
        "last_token_before_entity",
    ]
    n_splits_cv = 5

    for token_key in TOKEN_KEYS:
        print(f"\n=== Processing token key: {token_key} ===")

        # Vector extraction for the selected key
        X_all, y_all, _, titles_all = load_vectors_and_labels(
            hidden_states, meta_hidden_states, token_key, reduce="stack"
        )

        # ---------------------- hold-out separation -----------------
        X_train, X_test = X_all[train_idx], X_all[test_idx]
        y_train, y_test = y_all[train_idx], y_all[test_idx]
        groups_train = titles_all[train_idx]

        # … (any further model-training / CV code goes here) …


if __name__ == "__main__":
    main()
