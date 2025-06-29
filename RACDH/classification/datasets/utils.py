# utils.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shared helpers for SQuAD/WebQ evaluation."""

import json, os, sys
from pathlib import Path
import matplotlib as mpl
import joblib, matplotlib.pyplot as plt, numpy as np, torch
from scipy.stats import gaussian_kde
from tqdm import tqdm

sys.path.append(os.path.abspath("/home/ibrink/RACDH/RACDH/"))
from RACDH.classification.weighted_agg_predictor import StackHiddenStateClassifier
from RACDH.config import params
from RACDH.data_generation.inference.entity_tokens_find import _reconstruct_text, _sort_tokens, get_entity_span_text_align
from RACDH.data_generation.target_model import generate_completion_extract_hiddens, generate_completion
from RACDH.data_generation.utils.reading_data import load_json
from RACDH.data_generation.cross_encoder import get_similarity_score

# class Settings:
#     ROOT = Path(params.output_path)
#     MODEL = "weighted_agg_first_token_generation"  # overridable from CLI
#     MODEL_PATH = ROOT / "models" / f"{MODEL}.joblib"
#     TOKEN_KEY = "first_token_generation"
#     MAX_NEW_TOKENS = 10
#     PARAM_THRESHOLD = 0.5
#     PLOTS_DIR = ROOT / "plots"


# --------------------------------------------------------------------------- #
# QA‑pair abstractions                                                        #
# --------------------------------------------------------------------------- #

class _BaseQAPair:
    def __init__(self, question, answers, args):
        self.question = question.strip()
        self.answers = [a.strip() for a in answers if a.strip()]
        self.generated_answer = None
        self.generated_token_info = None
        self.args = args

    def _reconstruct_answer(self):
        toks = _sort_tokens(self.generated_token_info)
        self.generated_answer, _ = _reconstruct_text(toks)

    def evaluate_tokens(self, token_info):
        self.generated_token_info = token_info
        self._reconstruct_answer()
        spans = [get_entity_span_text_align(token_info, a, similarity_check_inference=False) for a in self.answers]
        if all(not s for s in spans):
            first_key = min(token_info, key=lambda k: k[0])
            return token_info[first_key], "Incorrect"
        for s in spans:
            if s:
                return s[self.args.token_key], "Correct"


_SQUAD_FEWSHOT = (
    "Context: Frank Herbert was an American science‑fiction author best known for his novel 'Dune.'\n"
    "Example Q: Who wrote “Dune”? A: Frank Herbert\n\n"
    "Context: Tokyo is the capital and most populous prefecture of Japan.\n"
    "Example Q: What is the capital of Japan? A: Tokyo\n\n"
)

_WEBQ_FEWSHOT = (
    "Question: Who wrote “Dune”?\nBrief answer: Frank Herbert\n\n"
    "Question: What is the capital of Japan?\nBrief answer: Tokyo\n\n"
)


class SquadPair(_BaseQAPair):
    def __init__(self, q, a, ctx, args):
        super().__init__(q, a, args)
        self.context = ctx.strip()

    @property
    def prompt(self):
        q = self.question.rstrip("?").capitalize() + "?"
        return f"{_SQUAD_FEWSHOT}Context: {self.context}\n\nQ: {q} A:"


class WebQPair(_BaseQAPair):
    @property
    def prompt(self):
        return f"{_WEBQ_FEWSHOT}Question: {self.question.capitalize()}\nBrief answer:"


# --------------------------------------------------------------------------- #
# Classifiers                                                                 #
# --------------------------------------------------------------------------- #

class HiddenStateClassifier:
    LABELS = {0: "Parametric", 1: "Contextual"}
    def __init__(self, path):
        self.pipe = joblib.load(path)
    def predict(self, vec):
        x = vec.to(torch.float32).cpu().numpy().reshape(1, -1)
        p_param, p_ctx = self.pipe.predict_proba(x)[0]
        return {"label": self.LABELS[int(p_ctx >= 0.5)], "p_parametric": float(p_param), "p_contextual": float(p_ctx)}


# --------------------------------------------------------------------------- #
# Loaders + KDE helper                                                        #
# --------------------------------------------------------------------------- #

def load_squad(path, args):
    raw = json.load(open(path, encoding="utf-8"))
    for art in raw.get("data", raw):
        for para in art.get("paragraphs", [art]):
            ctx = para["context"]
            for qa in para["qas"]:
                if qa.get("is_impossible"):
                    continue
                answers = list({a["text"] for a in qa["answers"]})  # remove duplicates
                yield SquadPair(qa["question"], answers, ctx, args)


def similar_enough(gold_answer, output):
    if gold_answer.lower() in output.lower():
        return True
    else:
        output_first_line = output.split('\n', 1)[0]
        score = get_similarity_score(gold_answer, output_first_line)
        if score > params.similarity_threshold_entity:
            print(f"Found similar answer: {gold_answer}-{output_first_line}")
            return True
        else:
            return False


def _find_entity_match_context(
    answers,
    samples,
    original_context,
):
    """Find a paragraph that *mentions* one of *answers* but does *not* have it in
    its own gold answer list.  Returns the paragraph text or *None*."""
    answers_lower = [a.lower() for a in answers]
    for other in samples:
        ctx = getattr(other, "context", None)
        if ctx is None or ctx == original_context:
            continue
        ctx_lower = ctx.lower()
        other_answers = [a.lower() for a in getattr(other, "answers", [])]
        for ans_l in answers_lower:
            if ans_l in ctx_lower and ans_l not in other_answers:
                return ctx
    return None

def load_webq(path, args):
    with open(str(path), encoding="utf-8") as f:
        data = json.load(f)
        for item in data:
            yield WebQPair(item["question"], item["answers"], args)


def kde_line(ax, series, label, colour, alpha=0.25):
    """
    Plot a KDE curve plus a translucent fill below it.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
    series : pandas.Series or 1-D array-like
    label : str         – legend label
    colour : str/tuple  – line & fill colour (hex or RGB)
    alpha : float       – fill opacity (0‒1); default 0.25
    """
    if len(series) < 2:
        return

    xs = np.linspace(0, 1, 300)
    kde_vals = gaussian_kde(series)(xs)

    # Line
    ax.plot(xs, kde_vals, label=label, color=colour, linewidth=1.8)

    # Shaded area
    ax.fill_between(xs, 0, kde_vals, color=colour, alpha=alpha, zorder=0)


def apply_kpmg_theme():

    BRAND = {
        "purple":      "#7214e6",
        "purple_light":"#b299ff",
        "blue":        "#1c47e3",
        "blue_dark":   "#00318d",
        "blue_light":  "#a0e7ff",
        "cyan":        "#00b8f9",
        "navy":        "#0e233e",
        "pink":        "#ff339a",
        "teal":        "#02bfa9",
    }

    mpl.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor":   "white",
        "axes.edgecolor":   BRAND["navy"],
        "axes.labelcolor":  BRAND["navy"],
        "xtick.color":      BRAND["navy"],
        "ytick.color":      BRAND["navy"],
        "grid.color":       "#d0d0d0",
        "font.family":      "sans-serif",
        "font.size":        12,
        "legend.frameon":   False,
        # Thin, on-brand grid
        "axes.grid":        True,
        "grid.linestyle":   ":",
        "grid.linewidth":   0.7,
    })


__all__ = [
    "Settings", "SquadPair", "WebQPair", "HiddenStateClassifier", "StackHiddenStateClassifier",
    "load_squad", "load_webq", "kde_line", "generate_completion_extract_hiddens", "generate_completion",
]