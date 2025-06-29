import sys
import os
import random
from tqdm import tqdm
sys.path.append(os.path.abspath("/home/ibrink/RACDH/RACDH/"))

from RACDH.data_generation.utils.reading_data import load_json
from RACDH.config import params
from RACDH.data_generation.target_model import generate_completion_extract_hiddens
from RACDH.data_generation.utils.print import *
from RACDH.data_generation.utils.writing_data import *
from RACDH.data_generation.inference.entity_tokens_find import (
    get_entity_span_text_align,
    reconstruct_generated_text
)
from collections import defaultdict
from RACDH.data_generation.instruct_model import generate_completion_GPT


if __name__ == "__main__":
    samples = load_json(
        f"{params.target_name}/{params.instruct_name}/completions_2.json")
    random.shuffle(samples)          # Contextual and Parametric are in order

    success = defaultdict(int)
    total   = defaultdict(int)

    # Holds hidden-state *stacks* for successful matches  ### UPDATED
    all_hidden_states = []

    # Metadata for each successful match
    data_to_save = []

    hidden_index = 0                 # only increments on success

    for sample in tqdm(samples, desc="Processing samples"):
        title   = sample["title"]
        passage = sample["passage"]
        entity  = sample["entity"]
        label   = sample["label"]

        print_h1(f"Title [{title}]")
        print_h2(label)
        print_h3(entity)
        print(passage)

        total[label] += 1

        # -------------------------------------------------
        # 1. Generate text and hidden-state stacks
        # -------------------------------------------------
        token_hiddens = generate_completion_extract_hiddens(
            prompt=passage,
            max_new_tokens=10,
            temperature=0.5,
            debug=False
        )

        # -------------------------------------------------
        # 2. Locate entity span
        # -------------------------------------------------
        entity_info = get_entity_span_text_align(token_hiddens, entity)



        if not entity_info:
            print_h2(f"No match found for entity: {entity}")
            continue

        # ------------- SUCCESS PATH ----------------------
        success[label] += 1
        print_h2(f"Match found for entity: {entity}")
        print("Entity tokens in the matched span:")
        for tok in entity_info["tokens"]:
            print(f"  Step {tok['step']}: {repr(tok['token_str'])}")

        # -------------------------------------------------
        # 3. Save hidden-state stacks                     ### UPDATED
        # -------------------------------------------------
        all_hidden_states.append({
            "first_token_entity":      entity_info["first_token_entity"],
            "last_token_entity":       entity_info["last_token_entity"],
            "first_token_generation":  entity_info["first_token_generation"],
            "last_token_before_entity": entity_info["last_token_before_entity"],
        })

        # -------------------------------------------------
        # 4. Save metadata
        # -------------------------------------------------
        data_to_save.append({
            "title": title,
            "passage": passage,
            "entity": entity,
            "generated": reconstruct_generated_text(token_hiddens),
            "label": label,
            "hidden_states_index": hidden_index,
            "similar_entity": entity_info["similar_entity"],
        })

        hidden_index += 1             # advance for next success

    # -----------------------------------------------------
    # 5. Persist tensors & metadata
    # -----------------------------------------------------
    write_tensors("hiddens_all_2.pt", all_hidden_states)
    write_to_json("hiddens_metadata_all_2.json", data_to_save)

    # -----------------------------------------------------
    # 6. Print success statistics
    # -----------------------------------------------------
    if total["contextual"]:
        print(f"Success rate CONTEXTUAL: {success['contextual'] / total['contextual']:.2%}")
    else:
        print("No contextual samples.")

    if total["parametric"]:
        print(f"Success rate PARAMETRIC: {success['parametric'] / total['parametric']:.2%}")
    else:
        print("No parametric samples.")

    if samples:
        overall = (success['contextual'] + success['parametric']) / len(samples)
        print(f"Success rate OVERALL: {overall:.2%}")
    else:
        print("No samples found.")
