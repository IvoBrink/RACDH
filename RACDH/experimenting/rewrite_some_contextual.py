import sys
import os
import json
import spacy
from tqdm import tqdm
sys.path.append(os.path.abspath("/home/ibrink/RACDH/RACDH/"))
from pathlib import Path
from RACDH.data_generation.utils.reading_data import load_json
from RACDH.data_generation.utils.writing_data import write_to_json
from RACDH.config import params
from RACDH.data_generation.completions.rewrite_contextual import rewrite_contextual_passage
from RACDH.data_generation.utils.print import *


def subtract_suffix(full_text: str, tail: str) -> str:
    """
    Remove *exactly one* final occurrence of `tail` from `full_text`.
    Raises if `tail` is **not** the very end of `full_text`.
    """
    if full_text.endswith(tail):
        return full_text.removesuffix(tail).strip()
    print_warning("Tail is not exactly part of the text")   # ⇐ built-in, removes once
    return None


# ── main ────────────────────────────────────────────────────────
if __name__ == "__main__":
    root = Path(f"{params.target_name}/{params.instruct_name}")
    samples = load_json(str(root / "completions.json"))
    counter = 0
    for s_1 in tqdm(samples, desc="Processing samples"):
        if s_1["label"] == "parametric" or "rewritten" in s_1.keys():
            continue
       
        if params.debug:
            print_h2(f"Rewrite passage for {s_1['title']}")
            print(s_1["original_passage"])

        rewritten_passage = None
        for s_2 in samples:
            if s_2["title"] == s_1["title"] and s_2["label"] == "parametric":
                if s_1["entity"].lower() in s_2["passage"].lower():
                    print_h3(f"Found parametric example for {s_1['entity']}")
                    rewritten_passage = subtract_suffix(s_2["passage"], s_2["appending_sentence"])
                    if rewritten_passage is not None:
                        rewritten_passage += " " + s_1["appending_sentence"]
                        s_1["rewritten"] = f"parametric found for {s_2['entity']}"
                        print(rewritten_passage)

        if rewritten_passage is None:
            print_h3(f"Defaulting to creating new rewritten example for {s_1['entity']}")
            rewritten_passage, removed_entity = rewrite_contextual_passage(s_1["original_passage"], s_1["entity"], s_1["appending_sentence"])
            if rewritten_passage is None:
                s_1["rewritten"] = "Not possible"
                rewritten_passage = s_1["passage"]
            else:
                s_1["rewritten"] = f"Newly generated for {removed_entity}"

        s_1["passage"] = rewritten_passage
        counter += 1
        if counter > 50:
            write_to_json("completions_rewrite.json", samples)
            counter = 0

    write_to_json("completions_rewrite.json", samples)