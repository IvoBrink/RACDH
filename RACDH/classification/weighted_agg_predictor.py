# weighted_agg_predictor.py
"""Universal wrapper that works with either the **linear** or the new **MLP** head.

The checkpoint itself decides which model class to instantiate:
* If any parameter name starts with "mlp." we assume it was trained with
  `WeightedAggMLP`.
* Otherwise we fall back to the classic `WeightedAggLogReg`.

This keeps your downstream inference code unchanged.
"""

from __future__ import annotations

import torch
from joblib import load
from typing import Any, Dict

# --- Model definitions ---------------------------------------------------------
# Always available (legacy linear head)
from RACDH.classification.all_layer_linear import WeightedAggLogReg, WeightedAggMLP


# -----------------------------------------------------------------------------
#  Predictor -------------------------------------------------------------------
# -----------------------------------------------------------------------------
class WeightedAggPredictor:
    """Light-weight inference-only wrapper for either model variant."""

    def __init__(self, joblib_path: str, device: str = "cpu") -> None:
        # ------------------------------------------------------------------
        # Load checkpoint payload
        # ------------------------------------------------------------------
        payload: Dict[str, Any] = load(joblib_path)

        self.token_key: str = payload["token_key"]
        self.device: str = device
        state_dict: Dict[str, torch.Tensor] = payload["state_dict"]

        # ------------------------------------------------------------------
        # Decide which model class to instantiate
        # ------------------------------------------------------------------
        is_mlp_ckpt = any(k.startswith("mlp.") for k in state_dict)
        if is_mlp_ckpt and WeightedAggMLP is not None:
            ModelClass = WeightedAggMLP  # type: ignore[assignment]
            model_kwargs = dict(
                num_layers=payload["num_layers"],
                hidden_dim=payload["hidden_dim"],
                bottleneck=payload.get("bottleneck", 64),
                dropout_p=0.0,  # irrelevant at inference time
            )
        else:  # ---- legacy linear head ------------------------------------
            ModelClass = WeightedAggLogReg
            model_kwargs = dict(
                num_layers=payload["num_layers"],
                hidden_dim=payload["hidden_dim"],
            )

        # Instantiate and load weights
        self.model = ModelClass(**model_kwargs)  # type: ignore[arg-type]
        self.model.load_state_dict(state_dict)
        self.model.to(self.device).eval()

    # ------------------------------------------------------------------
    #  Public API -------------------------------------------------------
    # ------------------------------------------------------------------
    @torch.inference_mode()
    def predict_proba(self, X_stack: torch.Tensor) -> torch.Tensor:
        """Return *P(class = 1)* for each sample.

        Parameters
        ----------
        X_stack : torch.Tensor
            Shape ``(N, L, H)`` — the same layer-stack used during training.
        """
        X_stack = X_stack.to(self.device, dtype=torch.float32)
        logits = self.model(X_stack)  # (N,)
        return torch.sigmoid(logits)

    def predict(self, X_stack: torch.Tensor, thresh: float = 0.5) -> torch.Tensor:
        """Return binary labels (0 = parametric, 1 = contextual)."""
        return (self.predict_proba(X_stack) > thresh).long()


# -----------------------------------------------------------------------------
#  Convenience wrapper used elsewhere in the codebase --------------------------
# -----------------------------------------------------------------------------
class StackHiddenStateClassifier:
    """A tiny helper that matches the old interface exactly."""

    LABELS = {0: "Parametric", 1: "Contextual"}

    def __init__(self, path: str):
        self.clf = WeightedAggPredictor(path)

    def predict(self, vec: torch.Tensor) -> Dict[str, Any]:
        if vec.ndim == 2:  # promote single sample to batch dim
            vec = vec.unsqueeze(0)

        p_ctx: float = float(self.clf.predict_proba(vec)[0])
        return {
            "label": self.LABELS[int(p_ctx >= 0.5)],
            "p_parametric": 1.0 - p_ctx,
            "p_contextual": p_ctx,
        }
