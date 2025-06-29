#!/usr/bin/env python
"""
Train layer‑weighted logistic regression / MLP on (L×H) hidden‑state stacks.
Now includes:
  • **Title‑aware validation split** identical to grid‑search
  • **Validation metrics (macro‑F1) printed each epoch**
  • **Early stopping** on validation F1 with configurable patience
"""

import os, sys, argparse, torch, numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from joblib import dump
import torch.nn as nn
from sparsemax import Sparsemax

# ------------------------------------------------------------------ #
#  project imports
# ------------------------------------------------------------------ #
sys.path.append(os.path.abspath("/home/ibrink/RACDH/RACDH/"))
from RACDH.config import params
from RACDH.data_generation.utils.reading_data import load_json
from RACDH.classification.utils import (
    groupwise_train_test_split,
    load_vectors_and_labels,
    seed_everything,
    make_title_aware_val_split
)

EPS = 1e-8

# ---------------------- Models ---------------------- #
class WeightedAggLogReg(nn.Module):
    def __init__(self, num_layers: int = 32, hidden_dim: int = 4096, dropout_p: float = 0.1):
        super().__init__()
        self.layer_logits = nn.Parameter(torch.zeros(num_layers))
        self.dropout = nn.Dropout(dropout_p)
        self.classifier = nn.Linear(hidden_dim, 1, bias=True)
        # self.sparsemax = Sparsemax(dim=0)

    def _norm(self, h: torch.Tensor) -> torch.Tensor:
        return h / (h.norm(dim=-1, keepdim=True) + EPS)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self._norm(hidden_states)
        # weights = self.sparsemax(self.layer_logits.unsqueeze(0)).squeeze(0)
        weights = torch.softmax(self.layer_logits, dim=0)
        agg = torch.einsum("blh,l->bh", hidden_states, weights)
        agg = self.dropout(agg)
        logits = self.classifier(agg)
        return logits.squeeze(-1)

class WeightedAggMLP(nn.Module):
    def __init__(self, num_layers: int = 32, hidden_dim: int = 4096, bottleneck: int = 64, dropout_p: float = 0.0):
        super().__init__()
        self.layer_logits = nn.Parameter(torch.zeros(num_layers))
        self.sparsemax = Sparsemax(dim=0)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, bottleneck, bias=False),
            nn.GELU(),
            nn.Dropout(dropout_p),
            nn.Linear(bottleneck, 1, bias=True),
        )

    def _norm(self, h: torch.Tensor) -> torch.Tensor:
        return h / (h.norm(dim=-1, keepdim=True) + EPS)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self._norm(hidden_states)
        weights = self.sparsemax(self.layer_logits.unsqueeze(0)).squeeze(0)
        agg = torch.einsum("blh,l->bh", hidden_states, weights)
        logits = self.mlp(agg)
        return logits.squeeze(-1)

# ---------------------- Main Training ---------------------- #
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--token_key", choices=[
        "first_token_entity", "last_token_entity", "first_token_generation", "last_token_before_entity"],
        default="first_token_generation")
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=3, help="early‑stopping patience (epochs)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--use_mlp", action="store_true")
    args = parser.parse_args()

    seed_everything()

    hidden_states = torch.load(os.path.join(params.output_path,
        f"{params.target_name}/{params.instruct_name}/hiddens_all_2.pt"))
    meta = load_json(f"{params.target_name}/{params.instruct_name}/hiddens_metadata_all_2.json")

    X, y, *_ = load_vectors_and_labels(hidden_states, meta, args.token_key, reduce="stack")
    num_layers, hidden_dim = X.shape[1:3]
    print(f"Loaded {len(X)} samples — stack shape = ({num_layers}, {hidden_dim})")

    print(f"Token key: {args.token_key}")
    train_idx, test_idx = groupwise_train_test_split(meta)

    inner_train_idx, val_idx = make_title_aware_val_split(
        train_idx=np.array(train_idx), meta=meta, val_frac=0.15, seed=42)

    X_train, y_train = X[inner_train_idx], y[inner_train_idx]
    X_val, y_val = X[val_idx], y[val_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    train_dl = DataLoader(TensorDataset(X_train, y_train), batch_size=args.batch, shuffle=True)

    if args.use_mlp:
        model_name = "mlp"
        model = WeightedAggMLP(num_layers=num_layers, hidden_dim=hidden_dim, bottleneck=64, dropout_p=0.1)
        lr = 1e-3
    else:
        model_name = "logreg"
        model = WeightedAggLogReg(num_layers=num_layers, hidden_dim=hidden_dim)
        lr = 2e-3
    model.to(args.device)

    pos = y_train.sum().item(); neg = len(y_train) - pos
    pos_weight = torch.tensor([neg / pos], device=args.device)
    crit = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-03)

    best_val_f1 = 0.0
    epochs_no_improve = 0
    spark = "▁▂▃▄▅▆▇█"
    for epoch in range(1, args.epochs + 1):
        model.train(); running_loss = 0.0; correct = 0
        for xb, yb in train_dl:
            xb, yb = xb.to(args.device), yb.to(args.device)
            optim.zero_grad()
            logits = model(xb)
            loss = crit(logits, yb.float())
            loss.backward(); optim.step()
            running_loss += loss.item() * xb.size(0)
            preds = (torch.sigmoid(logits) > 0.5)
            correct += (preds == yb).sum().item()
        train_acc = correct / len(train_dl.dataset)

        model.eval()
        with torch.no_grad():
            logits_val = model(X_val.to(args.device))
            preds_val = (torch.sigmoid(logits_val) > 0.5).cpu()
        _, _, f1_val, _ = precision_recall_fscore_support(
            y_val.numpy(), preds_val.numpy(), average="macro", zero_division=0)

        if model_name == "mlp": 
            w = model.sparsemax(model.layer_logits.unsqueeze(0).detach().cpu()).squeeze().numpy()
        else:
            w = torch.softmax(model.layer_logits.detach().cpu(), dim=0).numpy()
        w_min, w_max = w.min(), w.max()
        row = "".join(spark[int(((x - w_min) / (w_max - w_min + 1e-12)) * 7)] for x in w)

        print(f"Epoch {epoch:02d} | loss {running_loss/len(train_dl.dataset):.4f} | "
              f"train‑acc {train_acc:.3f} | val‑F1 {f1_val:.3f} | weights: {row}" )

        if f1_val > best_val_f1 + 1e-4:
            best_val_f1 = f1_val
            epochs_no_improve = 0
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= args.patience:
                print(f"Early stopping at epoch {epoch} — no improvement for {args.patience} epochs.")
                break

    model.load_state_dict(best_state)
    model.to(args.device).eval()
    print(f"Best validation F1: {best_val_f1:.3f}")

    with torch.no_grad():
        logits_test = model(X_test.to(args.device))
        y_hat = (torch.sigmoid(logits_test) > 0.5).cpu()

    test_acc = accuracy_score(y_test.numpy(), y_hat.numpy())
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_test.numpy(), y_hat.numpy(), labels=[0,1], zero_division=0)
    macro_f1 = (f1[0] + f1[1]) / 2
    print(f"Test accuracy {test_acc:.3f} • Macro‑F1 {macro_f1:.3f}\n" \
          f"Class 0 F1 {f1[0]:.3f} | Class 1 F1 {f1[1]:.3f}")

    MODEL_PATH = f"RACDH/data/models/{params.target_name}/weighted_agg_{model_name}_{args.token_key}.joblib"
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    payload = {"state_dict": {k: v.cpu() for k, v in best_state.items()},
               "num_layers": num_layers, "hidden_dim": hidden_dim, "token_key": args.token_key}
    dump(payload, MODEL_PATH)
    print(f"Saved best model to {MODEL_PATH}")

    if model_name == "mlp": 
        w = model.sparsemax(model.layer_logits.unsqueeze(0).detach().cpu()).squeeze().numpy()
    else:
        w = torch.softmax(model.layer_logits.detach().cpu(), dim=0).numpy()
    plot_dir = f"RACDH/data/plots/{params.target_name}"
    os.makedirs(plot_dir, exist_ok=True)
    weights_path = os.path.join(plot_dir, f"layer_weights_{model_name}_{args.token_key}.npy")
    np.save(weights_path, w)
    print(f"Saved layer weights to {weights_path}")

    BLUE_GRADIENT = ["#a0e7ff", "#1c47e3", "#00318d"]
    mpl.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor":   "white",
        "axes.edgecolor":   "#0e233e",
        "axes.labelcolor":  "#0e233e",
        "xtick.color":      "#0e233e",
        "ytick.color":      "#0e233e",
        "grid.color":       "#d0d0d0",
        "font.family":      "sans-serif",
        "font.size":        12,
    })

    cmap = LinearSegmentedColormap.from_list("kpmg_weight", BLUE_GRADIENT)
    norm = mpl.colors.Normalize(vmin=float(np.min(w)), vmax=float(np.max(w)))
    colors = cmap(norm(w))

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(
        x=range(len(w)),
        height=w,
        color=colors,
        edgecolor="#0e233e",
        linewidth=1,
    )

    ax.set_title(
        f"Weight per Transformer layer for {args.token_key}\n"
        f"Test acc {test_acc:.3f} • Contextual Recall {rec[1]:.3f}",
        loc="left", pad=14, color="#0e233e", weight="bold"
    )
    ax.set_xlabel("Layer index (0 = embedding, higher = deeper)")
    ax.set_ylabel("Learned aggregation weight")
    ax.spines[["top", "right"]].set_visible(False)
    ax.yaxis.grid(True, linestyle=":", linewidth=0.7)
    ax.set_axisbelow(True)

    sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, pad=0.02)
    cbar.set_label("Aggregation weight")

    plt.tight_layout()
    plt.savefig(f"{plot_dir}/layer_dist_{model_name}_{args.token_key}.png", dpi=300)
    plt.close(fig)

if __name__ == "__main__":
    main()
