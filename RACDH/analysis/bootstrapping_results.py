import argparse
import json
import sys, os
import numpy as np
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from RACDH.classification.weighted_agg_predictor import WeightedAggPredictor

MODEL_ALIASES = {
    "llama":   "Llama-3.1-8B",
    "mistral": "Mistral-7B-v0.1",
    "qwen":    "Qwen2.5-7B",
}

parser = argparse.ArgumentParser()
parser.add_argument("--model", default="llama", choices=list(MODEL_ALIASES),
                    help="Target model: llama | mistral | qwen")
parser.add_argument("--dataset", default="squad", help="Dataset subfolder, e.g. squad or webq")
parser.add_argument("--token_key", default="first_token_generation",
                    choices=["first_token_generation", "first_token_entity",
                             "last_token_entity", "last_token_before_entity"])
parser.add_argument("--classifier", default="logreg", choices=["logreg", "mlp"])
args = parser.parse_args()

model_name = MODEL_ALIASES[args.model]
BASE       = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))
DATA_PATH  = f"{BASE}/{args.dataset}/{model_name}/infer_{args.token_key}_balanced_2000.json"
MODEL_PATH = f"{BASE}/models/{model_name}/weighted_agg_{args.classifier}_{args.token_key}.joblib"

# ── Load data ──────────────────────────────────────────────────────────────
with open(DATA_PATH) as f:
    data = json.load(f)

print(f"Number of records : {len(data)}")
print(f"Fields            : {list(data[0].keys())}")
print(f"hidden shape      : {np.array(data[0]['hidden']).shape}  (layers x hidden_dim)")
print()

# ── Load probe ─────────────────────────────────────────────────────────────
probe = WeightedAggPredictor(MODEL_PATH)
print(f"Loaded probe  : {MODEL_PATH}")
print(f"  token_key={probe.token_key}")
print()

# ── Run predictions ────────────────────────────────────────────────────────
hiddens = torch.tensor(
    np.array([r["hidden"] for r in data], dtype=np.float32)
)  # (N, L, H)

# squad = all contextual, webq = all parametric
GROUND_TRUTH = {"squad": 1, "webq": 0}
if args.dataset not in GROUND_TRUTH:
    raise ValueError(f"Unknown dataset '{args.dataset}'; expected one of {list(GROUND_TRUTH)}")
gt_label = GROUND_TRUTH[args.dataset]
gt_name  = "contextual" if gt_label == 1 else "parametric"

p_contextual = probe.predict_proba(hiddens).numpy()
preds = probe.predict(hiddens).numpy()  # 0 = parametric, 1 = contextual

n = len(preds)
n_ctx = int(preds.sum())
print(f"Predictions : {n_ctx} contextual, {n - n_ctx} parametric out of {n}")

# Correct = prediction matches the ground-truth label for this dataset
correct = (preds == gt_label).astype(int)
accuracy = correct.mean()
print(f"Accuracy    : {accuracy:.1%}  (ground truth: all {gt_name})")

# ── Bootstrap CI ───────────────────────────────────────────────────────────
N_BOOT = 10_000
rng = np.random.default_rng(42)
boot_accs = np.array([
    correct[rng.integers(0, n, size=n)].mean()
    for _ in range(N_BOOT)
])
ci_lo, ci_hi = np.percentile(boot_accs, [2.5, 97.5])
half = (ci_hi - ci_lo) / 2
print(f"Bootstrap   : {accuracy:.1%} ± {half:.1%}  (95% CI: {ci_lo:.1%} – {ci_hi:.1%}, n_boot={N_BOOT})")
