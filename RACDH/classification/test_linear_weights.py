#!/usr/bin/env python
"""
Layer-weight analysis for token-provenance classification.

Run examples
------------
# both datasets, full size
python layer_weight_analysis.py

# down-sample to 8k each and force balanced loss
python layer_weight_analysis.py --max_samples 8000 --effective_pos 0.5
"""
import os, sys, argparse, random, torch, numpy as np
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from joblib import dump
import matplotlib.pyplot as plt

# ------------------------------------------------------------------ #
#  project imports
# ------------------------------------------------------------------ #
sys.path.append(os.path.abspath("/home/ibrink/RACDH/RACDH/"))
from RACDH.config import params
from RACDH.data_generation.utils.reading_data import load_json
from RACDH.classification.utils import (
    groupwise_train_test_split,
    load_vectors_and_labels,
)

# ---------------------- Model ---------------------- #
class WeightedAggLogReg(torch.nn.Module):
    def __init__(self, num_layers: int = 32, hidden_dim: int = 4096,
                 dropout_p: float = 0.2):
        super().__init__()
        self.layer_logits = torch.nn.Parameter(torch.zeros(num_layers))
        self.dropout      = torch.nn.Dropout(dropout_p)
        self.classifier   = torch.nn.Linear(hidden_dim, 1, bias=True)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        weights = torch.softmax(self.layer_logits, dim=0)      # (L,)
        agg = torch.einsum("blh,l->bh", hidden_states, weights)# (B,H)
        logits = self.classifier(self.dropout(agg))            # (B,1)
        return logits.squeeze(-1)

# -------------------------- helpers -------------------------------- #
def downsample(X, y, meta, max_samples, seed=0):
    """Return a random subset of size ≤ max_samples (stratified by y)."""
    if max_samples <= 0 or len(y) <= max_samples:
        return X, y, meta
    rng = np.random.default_rng(seed)
    idx_pos = np.where(y == 1)[0]
    idx_neg = np.where(y == 0)[0]

    n_pos = int(round(max_samples * len(idx_pos) / len(y)))
    n_neg = max_samples - n_pos
    keep = np.concatenate([
        rng.choice(idx_pos, n_pos, replace=False),
        rng.choice(idx_neg, n_neg, replace=False)
    ])
    rng.shuffle(keep)
    return X[keep], y[keep], [meta[i] for i in keep]

# ------------------------------------------------------------------ #
def train_one_dataset(version_suffix, args):
    """
    version_suffix: ''  → small
                     '_2' → big
    """
    tag = 'small' if version_suffix == '' else 'big'

    # -------------------------------------------------------------- #
    #  Load data
    # -------------------------------------------------------------- #
    hs_file   = f"{params.output_path}/{params.target_name}/{params.instruct_name}/hiddens_all{version_suffix}.pt"
    meta_file = f"{params.target_name}/{params.instruct_name}/hiddens_metadata_all{version_suffix}.json"

    hidden_states = torch.load(hs_file)
    meta = load_json(meta_file)


    


    X, y, _, _ = load_vectors_and_labels(hidden_states, meta,
                                         args.token_key, reduce="stack")

    # ---------- NEW: optional down-sampling ----------------------- #
    X, y, meta = downsample(X, y, meta, args.max_samples, seed=42)   ### NEW

    num_layers, hidden_dim = X.shape[1], X.shape[2]
    print(f"[dataset '{tag}'] loaded {len(y):,} samples → stack=({num_layers},{hidden_dim})")

    # -------------------------------------------------------------- #
    #  Train/test split
    # -------------------------------------------------------------- #
    train_idx, test_idx = groupwise_train_test_split(meta)
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    train_dl = DataLoader(TensorDataset(X_train, y_train),
                          batch_size=args.batch, shuffle=True)

    # -------------------------------------------------------------- #
    #  Model, optimiser, loss
    # -------------------------------------------------------------- #
    model = WeightedAggLogReg(num_layers=num_layers, hidden_dim=hidden_dim).to(args.device)
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    # ---------- NEW: loss re-weighting switch --------------------- #
    if args.effective_pos > 0:                         ### NEW
        pos_weight_val = (1 - args.effective_pos) / args.effective_pos
    else:                                              ### NEW (fallback = data driven)
        pos = y_train.sum().item()
        neg = len(y_train) - pos
        pos_weight_val = neg / pos

    crit = torch.nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([pos_weight_val], device=args.device)
    )

    #  entropy regulariser
    def ent(logits): p = torch.softmax(logits, 0); return -(p * p.log()).sum()
    alpha = 0.01

    # ---------------------- training loop ------------------------- #
    for epoch in range(1, args.epochs + 1):
        model.train(); running, correct = 0.0, 0
        for xb, yb in train_dl:
            xb, yb = xb.to(args.device), yb.to(args.device).float()
            optim.zero_grad()
            logits = model(xb)
            loss = crit(logits, yb) - alpha * ent(model.layer_logits)
            loss.backward(); optim.step()

            running += loss.item() * xb.size(0)
            correct += ((torch.sigmoid(logits) > .5) == yb.bool()).sum().item()

        if epoch % 1 == 0:
            print(f"  epoch {epoch:02d} | train-loss {running/len(train_dl.dataset):.4f}"
                  f" | train-acc {correct/len(train_dl.dataset):.3f}")

    # ---------------------- evaluation ---------------------------- #
    model.eval()
    with torch.no_grad():
        logits = model(X_test.to(args.device))
        y_hat  = (torch.sigmoid(logits) > 0.5).cpu()
    acc = accuracy_score(y_test, y_hat)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_test, y_hat, labels=[0,1], zero_division=0
    )
    print(f"  test-acc {acc:.3f} | contextual-recall {rec[1]:.3f}")

    # save weights for the comparison plot
    np.save(f"sample_layer_weights_{tag}.npy",
            torch.softmax(model.layer_logits, 0).detach().cpu().numpy())

    return tag, acc, prec, rec, f1

# ========================= main =================================== #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--token_key", default="first_token_entity",
                    choices=["first_token_entity","last_token_entity",
                             "first_token_generation","last_token_before_entity"])
    ap.add_argument("--versions", nargs="+", default=["small","big"],
                    choices=["small","big"])
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--device", type=str,
                    default="cuda" if torch.cuda.is_available() else "cpu")
    ### NEW flags -------------------------------------------------- #
    ap.add_argument("--max_samples", type=int, default=0,
                    help="cap dataset at N samples before split (0 = no cap)")
    ap.add_argument("--effective_pos", type=float, default=0.0,
                    help="desired positive-class prior for BCE re-weighting "
                         "(0 → use real imbalance)")
    args = ap.parse_args()

    results = {}
    for v in args.versions:
        suffix = '' if v == 'small' else '_2'
        res = train_one_dataset(suffix, args)
        results[res[0]] = res[1:]   # store metrics

    #  plotting & summary identical … (omitted for brevity)
    # ----------------------------------------------------------------- #
    small_weights = np.load("sample_layer_weights_small.npy")
    big_weights   = np.load("sample_layer_weights_big.npy")

    x = list(range(len(small_weights)))  # assume both have same number of layers

    plt.figure(figsize=(10, 6))
    plt.plot(x, small_weights, marker='o', label="small", linewidth=2)
    plt.plot(x, big_weights, marker='s', label="big", linewidth=2)

    plt.xlabel("Layer Index")
    plt.ylabel("Normalized Weight")
    plt.title("Layer-wise Weights: small vs. big")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("layer_weights_comparison.png")
    plt.show()

    print("\n--- Summary of Results ---")
    for tag, (acc, prec, rec, f1) in results.items():
        print(f"[{tag}] acc={acc:.3f}, prec={prec[1]:.3f}, rec={rec[1]:.3f}, f1={f1[1]:.3f}")


if __name__ == "__main__":
    main()