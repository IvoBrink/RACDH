import sys
import os
import random
sys.path.append(os.path.abspath("/home/ibrink/RACDH/RACDH/"))
from RACDH.config import params
from tqdm import tqdm
from RACDH.data_generation.utils.reading_data import load_json
from RACDH.data_generation.utils.print import *
from RACDH.data_generation.utils.writing_data import write_to_json
from RACDH.data_generation.completions.contextual_completion import add_contextual_completion
from RACDH.data_generation.completions.parametric_completion import add_parametric_completion
from RACDH.data_generation.completions.generic_completion import add_generic_completion


if __name__ == "__main__":
    samples_known = load_json(f"{params.target_name}/{params.instruct_name}/rewritten_known.json")
    samples = load_json(f"{params.target_name}/{params.instruct_name}/knowledge.json")
    existing_completions = load_json(f"{params.target_name}/{params.instruct_name}/completions.json")
    existing_titles_entities = [(completion["title"], completion["entity"]) for completion in existing_completions]
    current_batch = []  # Rename to make it clear this is just the current batch

    # Parametric
    if params.debug: print_h1("Generating Parametric completion examples")
    for sample_known in tqdm(samples_known, desc="Processing Parametric samples"):
        title = sample_known["title"]
        passage = sample_known["passage"]
        rewritten_passages = sample_known["rewritten_passages"]
        original_passage = sample_known["passage"]

        if len(rewritten_passages) == 0:
            continue
        if params.debug: print_h2(f"Passage [{title}]")

        for entity, rewritten_passage in rewritten_passages.items():
            if params.debug: print_h3(f"Entity [{entity}]")

            if (title, entity) in existing_titles_entities:
                if params.debug: print_h3(f"Data already exists!")
                continue

            completion = add_generic_completion(entity, original_passage)
            final_passage = None if completion is None else rewritten_passage + " " + completion

            if final_passage is not None:
                current_batch.append({
                    "title" : title,
                    "passage" : final_passage,
                    "entity" : entity,
                    "label" : "parametric",
                    "appending_sentence" : completion,
                    "original_passage" : passage
                })
                if len(current_batch) >= 10:
                    write_to_json("completions.json", current_batch, overwrite=False)
                    current_batch = []  # Reset after writing


    # Contextual
    if params.debug: print_h1("Generating Contextual completion examples")
    for sample in tqdm(samples, desc="Processing Contextual samples"):
        title = sample["title"]
        passage = sample["passage"] #TODO: get a passage from rewritten to mitigate removal bias, if it does not exist, create one.
        ignored_entities = sample["ignored_entities"]
        if len(ignored_entities) == 0:
            continue
        if params.debug: print_h2(f"Passage [{title}]")

        for entity in ignored_entities:
            if params.debug: print_h3(f"Entity [{entity}]")

            if (title, entity) in existing_titles_entities:
                if params.debug: print_h3(f"Data already exists!")
                continue

            completion = add_generic_completion(entity, passage)
            final_passage = None if completion is None else passage + " " + completion

            if final_passage is not None:
                current_batch.append({
                    "title" : title,
                    "passage" : final_passage,
                    "entity" : entity,
                    "label" : "contextual",
                    "original_passage" : passage
                })
                if len(current_batch) >= 10:
                    write_to_json("completions.json", current_batch, overwrite=False)
                    current_batch = []  # Reset after writing

    # Write any remaining items in the final batch
    if current_batch:
        write_to_json("completions.json", current_batch, overwrite=False)

    # After you're done with the model
    import torch
    torch.cuda.empty_cache()

