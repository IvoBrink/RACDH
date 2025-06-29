#!/usr/bin/env python
"""
Universal grid-search for the *layer-weighted* aggregation classifier.
"""
from __future__ import annotations

import argparse, itertools, json, os, sys, torch
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
from joblib import dump
from sklearn.metrics import accuracy_score, f1_score

from torch.utils.data import DataLoader, TensorDataset

# Project imports
sys.path.append(os.path.abspath("/home/ibrink/RACDH/RACDH/"))
from RACDH.config import params
from RACDH.data_generation.utils.reading_data import load_json
from RACDH.classification.utils import (
    groupwise_train_test_split,
    load_vectors_and_labels,
    seed_everything,
    make_title_aware_val_split
)
from RACDH.classification.all_layer_linear import WeightedAggLogReg, WeightedAggMLP

def build_model(
    *,
    num_layers: int,
    hidden_dim: int,
    dropout_p: float,
    bottleneck: int,
    use_mlp: bool,
) -> nn.Module:
    if use_mlp:
        if WeightedAggMLP is None:
            raise RuntimeError("WeightedAggMLP not available")
        return WeightedAggMLP(
            num_layers=num_layers,
            hidden_dim=hidden_dim,
            bottleneck=bottleneck,
            dropout_p=dropout_p,
        )
    return WeightedAggLogReg(
        num_layers=num_layers,
        hidden_dim=hidden_dim,
        dropout_p=dropout_p,
    )

def run_one_setting(
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_val: torch.Tensor,
    y_val: torch.Tensor,
    *,
    num_layers: int,
    hidden_dim: int,
    device: str,
    dropout_p: float,
    weight_decay: float,
    bottleneck: int,
    use_mlp: bool,
    lr: float = 1e-3,
    batch: int = 64,
    epochs: int = 10,
):
    model = build_model(
        num_layers=num_layers,
        hidden_dim=hidden_dim,
        dropout_p=dropout_p,
        bottleneck=bottleneck,
        use_mlp=use_mlp,
    ).to(device)

    pos, neg = y_train.sum().item(), len(y_train) - y_train.sum().item()
    crit = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([neg / pos], device=device))
    optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    train_dl = DataLoader(TensorDataset(X_train, y_train), batch_size=batch, shuffle=True)

    for _ in range(epochs):
        model.train()
        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            optim.zero_grad()
            logits = model(xb)
            loss = crit(logits, yb.float())
            loss.backward(); optim.step()

    model.eval()
    with torch.no_grad():
        logits = model(X_val.to(device))
        y_pred = (torch.sigmoid(logits) > 0.5).cpu()
    acc = accuracy_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred)
    weights = torch.softmax(model.layer_logits, 0).detach().cpu().numpy()
    return acc, f1, model, weights

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--token_key", default="first_token_generation", choices=[
        "first_token_entity", "last_token_entity", "first_token_generation", "last_token_before_entity"
    ])
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--val_split", type=float, default=0.15)
    parser.add_argument("--use_mlp", action="store_true")
    parser.add_argument("--bottleneck", type=int, default=128)
    args = parser.parse_args()

    seed_everything()
    hidden_states = torch.load(os.path.join(
        params.output_path,
        f"{params.target_name}/{params.instruct_name}/hiddens_all_2.pt",
    ))
    meta = load_json(f"{params.target_name}/{params.instruct_name}/hiddens_metadata_all_2.json")

    X, y, *_ = load_vectors_and_labels(hidden_states, meta, args.token_key, reduce="stack")
    num_layers, hidden_dim = X.shape[1:3]

    train_idx, test_idx = groupwise_train_test_split(meta)
    X_train_full, y_train_full = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx].to(args.device), y[test_idx].to(args.device)

    inner_train_idx, val_idx = make_title_aware_val_split(
        train_idx=np.array(train_idx), meta=meta, val_frac=args.val_split
    )
    X_train, y_train = X[inner_train_idx], y[inner_train_idx]
    X_val, y_val = X[val_idx], y[val_idx]

    grid: Dict[str, List[Any]] = {
        "dropout_p": [0, 0.1, 0.2],
        "weight_decay": [0, 5e-4, 1e-3, 2e-3],
        "lr": [5e-4, 1e-3, 2e-3],
        "bottleneck": [64],
    }
    bottleneck_space = grid["bottleneck"] if args.use_mlp else [args.bottleneck]
    results: List[Dict[str, Any]] = []
    spark = "▁▂▃▄▅▆▇█"

    for dropout_p, wd, lr, bottleneck in itertools.product(
        grid["dropout_p"],
        grid["weight_decay"],
        grid["lr"],
        bottleneck_space,
    ):
        acc, f1, model, w = run_one_setting(
            X_train, y_train, X_val, y_val,
            num_layers=num_layers,
            hidden_dim=hidden_dim,
            device=args.device,
            dropout_p=dropout_p,
            weight_decay=wd,
            bottleneck=bottleneck,
            use_mlp=args.use_mlp,
            epochs=args.epochs,
            lr=lr,
        )
        results.append({
            "dropout_p": dropout_p,
            "weight_decay": wd,
            "lr": lr,
            "bottleneck": bottleneck,
            "val_acc": acc,
            "val_f1": f1,
            "model": model.cpu().eval(),
            "weights": w,
        })

        w_min, w_max = w.min(), w.max()
        row = "".join(spark[int(((x - w_min) / (w_max - w_min + 1e-12)) * 7)] for x in w)
        print(f"[grid] d={dropout_p:.2f} wd={wd:.1e} lr={lr:.0e} bn={bottleneck:<3} → "
              f"acc {acc:.3f} f1 {f1:.3f} layers:{row}")

    best = max(results, key=lambda r: (r["val_f1"], r["val_acc"]))
    print("\n=== Best hyper-parameters ===")
    print(json.dumps({k: best[k] for k in [
        "dropout_p", "weight_decay", "lr", "bottleneck"
    ]}, indent=2))

if __name__ == "__main__":
    main()
