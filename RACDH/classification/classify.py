import os
import sys
import torch
import numpy as np
from collections import defaultdict

# For headless plotting on HPC:
import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.model_selection import GroupKFold, cross_val_predict, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Adjust to your local paths if needed
sys.path.append(os.path.abspath("/home/ibrink/RACDH/RACDH/"))
from RACDH.config import params
from RACDH.data_generation.utils.reading_data import load_json

def load_vectors_and_labels(hidden_states, meta_hidden_states, token_key="first_token_hidden"):
    all_vectors = []
    all_labels = []
    similar_entity_flags = []
    titles = []

    for sample in meta_hidden_states:
        label_str = sample["label"]
        hidden_idx = sample["hidden_states_index"]
        hiddens = hidden_states[hidden_idx]

        token_vec = hiddens[token_key].to(torch.float32).numpy()
        all_vectors.append(token_vec)
        all_labels.append(1 if label_str == "contextual" else 0)
        similar_entity_flags.append(sample.get("similar_entity", False))
        titles.append(sample.get("title", ""))

    all_vectors = np.array(all_vectors).reshape(-1, 4096)
    all_labels = np.array(all_labels, dtype=int)
    similar_entity_flags = np.array(similar_entity_flags, dtype=bool)

    return all_vectors, all_labels, similar_entity_flags, np.array(titles)

def groupwise_train_test_split(meta_hidden_states, test_size=0.2, random_state=42):
    title_to_indices = defaultdict(list)
    for idx, sample in enumerate(meta_hidden_states):
        title = sample.get("title", "")
        title_to_indices[title].append(idx)

    unique_titles = list(title_to_indices.keys())
    train_titles, test_titles = train_test_split(unique_titles, test_size=test_size, random_state=random_state)

    train_indices = [i for t in train_titles for i in title_to_indices[t]]
    test_indices = [i for t in test_titles for i in title_to_indices[t]]

    return train_indices, test_indices

def run_dimensionality_reduction(X, method="pca", n_components=2):
    if method == "pca":
        reducer = PCA(n_components=n_components)
        X_2d = reducer.fit_transform(X)
    else:
        raise ValueError(f"Unknown dimensionality reduction method: {method}")

    return X_2d

def plot_2d_scatter(X_2d, labels, title, outfile):
    idx_0 = (labels == 0)
    idx_1 = (labels == 1)

    fig, ax = plt.subplots()
    ax.scatter(X_2d[idx_0, 0], X_2d[idx_0, 1], marker="o", label="Parametric (0)")
    ax.scatter(X_2d[idx_1, 0], X_2d[idx_1, 1], marker="x", label="Contextual (1)")

    ax.set_title(title)
    ax.set_xlabel("Dim 1")
    ax.set_ylabel("Dim 2")
    ax.legend()

    plt.savefig(outfile)
    plt.close(fig)

def classify_and_report_cv(X_2d, labels, similar_entity_flags=None, groups=None, n_splits=5):
    if groups is None:
        raise ValueError("You must provide `groups` (e.g., titles) to avoid data leakage.")

    gkf = GroupKFold(n_splits=n_splits)
    clf = LogisticRegression()
    y_pred = cross_val_predict(clf, X_2d, labels, cv=gkf, groups=groups)

    acc = accuracy_score(labels, y_pred)
    print(f"GroupKFold Accuracy ({n_splits}-fold): {acc:.3f}")
    print(
        classification_report(
            labels,
            y_pred,
            target_names=["Parametric (0)", "Contextual (1)"],
            zero_division=0
        )
    )

    if similar_entity_flags is not None and np.any(similar_entity_flags):
        similar_entity_indices = np.where(similar_entity_flags)[0]
        similar_entity_labels = labels[similar_entity_indices]
        similar_entity_preds = y_pred[similar_entity_indices]

        similar_entity_acc = accuracy_score(similar_entity_labels, similar_entity_preds)
        print(f"Accuracy for samples with similar_entity=True: {similar_entity_acc:.3f}")

        print("\nClassification Report for samples with similar_entity=True:")
        print(
            classification_report(
                similar_entity_labels,
                similar_entity_preds,
                target_names=["Parametric (0)", "Contextual (1)"],
                zero_division=0
            )
        )

if __name__ == "__main__":
    hidden_states = torch.load(os.path.join(params.output_path, f"{params.target_name}/{params.instruct_name}/hiddens_rewrite.pt"))
    meta_hidden_states = load_json(f"{params.target_name}/{params.instruct_name}/hiddens_metadata_rewrite.json")

    train_indices, test_indices = groupwise_train_test_split(meta_hidden_states)

    token_keys = ["first_token_hidden", "last_token_hidden"]
    methods = ["pca"]

    for token_key in token_keys:
        print(f"\n=== Processing {token_key} ===")

        X_all, labels_all, flags_all, titles_all = load_vectors_and_labels(hidden_states, meta_hidden_states, token_key)

        selected_indices = train_indices + test_indices
        X = X_all[selected_indices]
        labels = labels_all[selected_indices]
        similar_entity_flags = flags_all[selected_indices]
        titles = titles_all[selected_indices]

        print("Collected hidden states shape:", X.shape)
        print("Labels shape:", labels.shape)
        print(f"Number of samples with similar_entity=True: {np.sum(similar_entity_flags)}")

        for method in methods:
            print(f"\n--- Dim Reduction Method: {method.upper()} ---")
            X_2d = run_dimensionality_reduction(X, method=method, n_components=50)

            plot_title = f"{method.upper()} (2D) of LLM Hidden States [{token_key}]"
            outfile = os.path.join(params.output_path + "plots/", f"{method}_{token_key}_plot.pdf")
            # plot_2d_scatter(X_2d, labels, plot_title, outfile)
            # print(f"Saved 2D plot to {outfile}")

            classify_and_report_cv(X_2d, labels, similar_entity_flags, groups=titles)