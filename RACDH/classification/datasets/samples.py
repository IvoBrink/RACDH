#!/usr/bin/env python3
"""
Generate a balanced (2 000 correct / 2 000 incorrect) evaluation set
for SQuAD or WebQuestions.

Key refactors

*   Factorised repeated logic into small helpers
*   Added type hints and constants for readability
*   Centralised all ‘mode’ handling (contextswitch, prioronly, etc.)
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List

from tqdm import tqdm

from utils import (
    _find_entity_match_context,
    generate_completion,
    generate_completion_extract_hiddens,
    load_squad,
    load_webq,
    similar_enough,
)

# Add project root for RACDH config
sys.path.append(os.path.abspath("/home/ibrink/RACDH/RACDH/"))
from RACDH.config import params

#  Constants                                                                   #
TARGET_PER_CLASS = 2_000
TAG_SUFFIX = {
    "all_context_switched": "_parametric",
    "only_prior_no_switch": "_prior_no_switch",
    "only_prior_entity_decoy": "_prior_entity_decoy",
}

#  CLI                                                                        
def parse_args() > argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate a balanced (2 k/2 k) evaluation set (SQuAD / WebQ)."
    )
    p.add_argument("dataset", default="squad", help="'squad', 'webq', or a JSON file")
    grp = p.add_mutually_exclusive_group()
    grp.add_argument("allcontextswitched", action="store_true")
    grp.add_argument("onlypriornoswitch", action="store_true")
    grp.add_argument("onlypriorentitydecoy", action="store_true")
    p.add_argument("seed", type=int, default=0)
    p.add_argument(
        "tokenkey",
        choices=[
            "first_token_entity",
            "last_token_entity",
            "first_token_generation",
            "last_token_before_entity",
        ],
        default="first_token_generation",
        help="Which hiddenstate stack to use",
    )
    return p.parse_args()


def load_dataset(src: str, cli: argparse.Namespace):
    """Return (iterator, canonical dataset name)."""
    std_names = {"squad", "webq", "webquestions"}
    if src.lower() in std_names:
        name = "webq" if "webq" in src.lower() else "squad"
        loader = load_webq if name == "webq" else load_squad
        file_name = "train.json" if name == "webq" else "trainv2.0.json"
        path = Path(params.output_path) / name / file_name
        return loader(path, cli), name

    # custom JSON path
    path = Path(src)
    if not path.exists():
        raise FileNotFoundError(path)
    loader = load_webq if "webq" in path.name.lower() else load_squad
    return loader(path, cli), ("webq" if loader is load_webq else "squad")


def hidden_and_label(sample, cli, ds_name: str):
    toks = generate_completion_extract_hiddens(sample.prompt, max_new_tokens=10)
    h_vec, label = sample.evaluate_tokens(toks)

    # Extra verification (SQuAD only, firsttokengeneration only)
    if (
        label == "Incorrect"
        and cli.token_key == "first_token_generation"
        and ds_name == "squad"
        and any(similar_enough(a, sample.generated_answer) for a in sample.answers)
    ):
        label = "Correct"

    h_last = h_vec[1] if isinstance(h_vec, (list, tuple)) else h_vec
    return (h_last.tolist() if hasattr(h_last, "tolist") else h_last), label


def record_sample(
    sample,
    hidden,
    label: str,
    counts: Dict[str, int],
    results: List[Dict[str, Any]],
    context_keep: str | None = None,
) > bool:
    """Store a sample iff its class bucket isn’t full."""
    if counts[label] >= TARGET_PER_CLASS:
        return False

    counts[label] += 1
    results.append(
        {
            "question": sample.question,
            "context": context_keep,
            "answer": sample.generated_answer,
            "correct_answers": sample.answers,
            "answer_correct": label == "Correct",
            "hidden": hidden,
        }
    )
    return True


def main() > None:
    cli = parse_args()
    random.seed(cli.seed)


    samples_iter, ds_name = load_dataset(cli.dataset, cli)
    samples = list(samples_iter)
    random.shuffle(samples)

    prior_mode = cli.only_prior_no_switch or cli.only_prior_entity_decoy
    if ds_name == "webq" and prior_mode:
        raise ValueError("onlyprior* modes are valid for SQuAD only.")

    all_contexts = (
        [getattr(s, "context", None) for s in samples if getattr(s, "context", None)]
        if cli.all_context_switched or cli.only_prior_entity_decoy
        else []
    )

    counts = {"Correct": 0, "Incorrect": 0}
    results: list[Dict[str, Any]] = []
    prior_skipped = 0

    with tqdm(samples, desc="Generating", unit="sample") as bar:
        for s in bar:
            #SQuADspecific “prior knowledge” probe 
            if ds_name == "squad" and not cli.all_context_switched:
                probe_q = (
                    "Q: Who wrote the book 1984?\nA: George Orwell\nQ: "
                    f"{s.question}\nA:"
                )
                _, prior_out = generate_completion(probe_q, s.answers[0], max_new_tokens=10)

                prior_hit = any(similar_enough(a, prior_out) for a in s.answers)
                if prior_hit:
                    prior_skipped += 1

                    if cli.only_prior_no_switch:
                        hidden, label = hidden_and_label(s, cli, ds_name)
                        if record_sample(
                            s,
                            hidden,
                            label,
                            counts,
                            results,
                            s.context,
                        ):
                            if sum(counts.values()) >= TARGET_PER_CLASS:
                                    break
                            bar.set_postfix(correct=counts["Correct"], incorrect=counts["Incorrect"])
                        continue

                    if cli.only_prior_entity_decoy:
                        new_ctx = _find_entity_match_context(s.answers, samples, s.context)
                        if new_ctx is None:
                            continue
                        s.context = new_ctx
                        hidden, label = hidden_and_label(s, cli, ds_name)
                        if record_sample(
                            s,
                            hidden,
                            label,
                            counts,
                            results,
                            s.context,
                        ):
                            if sum(counts.values()) >= TARGET_PER_CLASS:
                                    break
                            bar.set_postfix(correct=counts["Correct"], incorrect=counts["Incorrect"])
                        continue

                    #  Not in a “prioronly” mode → discard and move on
                    continue
                elif (cli.only_prior_no_switch or cli.only_prior_entity_decoy):
                    continue
            # Context swap (parametric)
            if cli.all_context_switched:
                s.context = random.choice(all_contexts)

            #  Normal evaluation path 
            hidden, label = hidden_and_label(s, cli, ds_name)
            if record_sample(
                s,
                hidden,
                label,
                counts,
                results,
                None if ds_name != "squad" else s.context,
            ):
                bar.set_postfix(correct=counts["Correct"], incorrect=counts["Incorrect"])

            #  Earlyexit criteria 
            if all(v >= TARGET_PER_CLASS for v in counts.values()):
                break
            if ds_name == "webq" and counts["Correct"] >= TARGET_PER_CLASS:
                break


    #  Persist                                                               
    tag = f"{cli.token_key}_balanced_{TARGET_PER_CLASS}"
    for flag, suffix in TAG_SUFFIX.items():
        if getattr(cli, flag):
            tag += suffix
            break
    

    out_dir = Path(params.output_path) / ds_name / params.target_name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"infer_{tag}.json"
    out_path.write_text(json.dumps(results, indent=2))

    #  Report                                                                #
    print(f"\n✓  Finished. Saved {len(results)} records to {out_path.resolve()}")
    print("Counts →", counts)
    if prior_skipped:
        print(f"Skipped {prior_skipped} SQuAD questions due to priorknowledge detection.")


if __name__ == "__main__":
    main()
