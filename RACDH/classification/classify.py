import os
import sys
import torch
import numpy as np

# For headless plotting on HPC:
import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

from pacmap import PaCMAP

# Adjust to your local paths if needed
sys.path.append(os.path.abspath("/home/ibrink/RACDH/RACDH/"))
from RACDH.config import params
from RACDH.data_generation.utils.reading_data import load_json


def load_vectors_and_labels(hidden_states, meta_hidden_states, token_key="first_token_hidden"):
    """
    Loads either 'first_token_hidden' or 'last_token_hidden' from hidden_states,
    returns (all_vectors, all_labels) as NumPy arrays.
    
    Label 1 => "contextual", Label 0 => "parametric"
    """
    all_vectors = []
    all_labels = []

    for sample in meta_hidden_states:
        label_str = sample["label"]  # e.g., "contextual" or "parametric"
        hidden_idx = sample["hidden_states_index"]
        hiddens = hidden_states[hidden_idx]

        # Extract the chosen token's vector, cast to float32, then to NumPy
        token_vec = hiddens[token_key].to(torch.float32).numpy()  # shape [4096]
        all_vectors.append(token_vec)
        # Convert label to 0 or 1 (0=parametric, 1=contextual)
        all_labels.append(1 if label_str == "contextual" else 0)

    all_vectors = np.array(all_vectors).reshape(-1, 4096)
    all_labels = np.array(all_labels, dtype=int)

    return all_vectors, all_labels


def run_dimensionality_reduction(X, method="pca", n_components=2):
    """
    Given data X (num_samples x 4096), reduce to n_components dims using
    either PCA or PaCMAP.
    """
    if method == "pca":
        reducer = PCA(n_components=n_components)
        X_2d = reducer.fit_transform(X)
    elif method == "pacmap":
        reducer = PaCMAP(n_components=n_components)
        # Using 'pca' init often leads to better results
        X_2d = reducer.fit_transform(X, init="pca")
    else:
        raise ValueError(f"Unknown dimensionality reduction method: {method}")

    return X_2d


def plot_2d_scatter(X_2d, labels, title, outfile):
    """
    Plot a 2D scatter given X_2d (num_samples x 2) and labels (0 or 1).
    Saves the figure to outfile.
    
    Label 0 => Parametric
    Label 1 => Contextual
    """
    idx_0 = (labels == 0)  # parametric
    idx_1 = (labels == 1)  # contextual

    fig, ax = plt.subplots()
    ax.scatter(X_2d[idx_0, 0], X_2d[idx_0, 1], marker="o", label="Parametric (0)")
    ax.scatter(X_2d[idx_1, 0], X_2d[idx_1, 1], marker="x", label="Contextual (1)")

    ax.set_title(title)
    ax.set_xlabel("Dim 1")
    ax.set_ylabel("Dim 2")
    ax.legend()

    plt.savefig(outfile)
    plt.close(fig)


def classify_and_report(X_2d, labels):
    """
    Train a logistic regression on X_2d -> labels, then print accuracy & classification report.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X_2d, labels, test_size=0.2, random_state=42
    )
    clf = LogisticRegression()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print("Accuracy on test set:", acc)
    # Pass zero_division=0 to suppress the ill-defined warnings:
    print(
        classification_report(
            y_test,
            y_pred,
            target_names=["Parametric (0)", "Contextual (1)"],
            zero_division=0
        )
    )



if __name__ == "__main__":
    # 1. Load hidden states
    hidden_states = torch.load(os.path.join(params.output_path, f"{params.target_name}/{params.instruct_name}/hiddens.pt"))
    meta_hidden_states = load_json(f"{params.target_name}/{params.instruct_name}/hiddens_metadata.json")

    # 2. We will do 2 runs: one for "first_token_hidden", one for "last_token_hidden"
    token_keys = ["first_token_hidden", "last_token_hidden"]

    # 3. We'll run both PCA & PaCMAP
    methods = ["pca", "pacmap"]

    for token_key in token_keys:
        print(f"\n=== Processing {token_key} ===")
        X, labels = load_vectors_and_labels(hidden_states, meta_hidden_states, token_key)
        print("Collected hidden states shape:", X.shape)
        print("Labels shape:", labels.shape)

        for method in methods:
            print(f"\n--- Dim Reduction Method: {method.upper()} ---")

            # 4. Dimensionality Reduction
            X_2d = run_dimensionality_reduction(X, method=method, n_components=50)

            # 5. Plot
            plot_title = f"{method.upper()} (2D) of LLM Hidden States [{token_key}]"
            outfile = os.path.join(params.output_path + "plots/", f"{method}_{token_key}_plot.pdf")
            # plot_2d_scatter(X_2d, labels, plot_title, outfile)
            # print(f"Saved 2D plot to {outfile}")

            # 6. Classification
            classify_and_report(X_2d, labels)
