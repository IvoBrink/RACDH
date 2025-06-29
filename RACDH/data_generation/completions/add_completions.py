import sys
import os
import random


sys.path.append(os.path.abspath("/home/ibrink/RACDH/RACDH/"))
from RACDH.config import params
from tqdm import tqdm
from RACDH.data_generation.utils.reading_data import load_json
from RACDH.data_generation.utils.print import *
from RACDH.data_generation.utils.writing_data import write_to_json
from RACDH.data_generation.completions.generic_completion import add_generic_completion
from RACDH.data_generation.completions.rewrite_contextual import rewrite_contextual_passage


if __name__ == "__main__":
    samples_known = load_json(f"{params.target_name}/{params.instruct_name}/rewritten_known_2.json")
    samples = load_json(f"{params.target_name}/{params.instruct_name}/knowledge_2.json")
    existing_completions = load_json(f"{params.target_name}/{params.instruct_name}/completions_2.json")
    existing_titles_entities = [(completion["title"], completion["entity"]) for completion in existing_completions]
    current_batch = []

    dict_different_target = { (s["title"], s["entity"]): s["appending_sentence"] for s in load_json("Llama-3.1-8B/gpt-4o-mini/completions_2.json") }


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

            key = (title, entity)
            appending_sentence = dict_different_target.get(key)
            if appending_sentence:
                completion = appending_sentence
            else:
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
                    write_to_json("completions_2.json", current_batch, overwrite=False)
                    current_batch = []  # Reset after writing


    # Contextual
    if params.debug: print_h1("Generating Contextual completion examples")
    for sample in tqdm(samples, desc="Processing Contextual samples"):
        title = sample["title"]
        passage = sample["passage"]

        ignored_entities = sample["ignored_entities"]
        if len(ignored_entities) == 0:
            continue
        if params.debug: print_h2(f"Passage [{title}]")

        for entity in ignored_entities:
            if params.debug: print_h3(f"Entity [{entity}]")

            if (title, entity) in existing_titles_entities:
                if params.debug: print_h3(f"Data already exists!")
                continue


            key = (title, entity)
            appending_sentence = dict_different_target.get(key)
            if appending_sentence:
                completion = appending_sentence
            else:
                completion = add_generic_completion(entity, original_passage)

            final_passage = None if completion is None else passage + " " + completion

            if final_passage is not None:

                ### get rewritten passage from samples_known or generate new one
                rewritten_passage = None
                for sample_known in samples_known:
                    if sample_known["title"] == title:
                        for entity_s2, rewritten_passage_s2 in sample_known["rewritten_passages"].items():
                            if entity.lower() in rewritten_passage_s2.lower():
                                print_h3(f"Found parametric example for {entity}")
                                rewritten_passage = rewritten_passage_s2 + " " + completion
                                rewritten_meta = f"parametric found for {entity_s2}"
                                break
                        break
                if rewritten_passage is None:
                    print_h3(f"Defaulting to creating new rewritten example for {entity}")
                    rewritten_passage, removed_entity = rewrite_contextual_passage(passage, entity, completion)
                    if rewritten_passage is None:
                        rewritten_meta = "Not possible"
                        rewritten_passage = final_passage
                    else:
                        rewritten_meta = f"Newly generated for {removed_entity}"
                ####


                current_batch.append({
                    "title" : title,
                    "passage" : rewritten_passage,
                    "entity" : entity,
                    "label" : "contextual",
                    "appending_sentence" : completion,
                    "original_passage" : passage,
                    "rewritten" : rewritten_meta
                })
                if len(current_batch) >= 10:
                    write_to_json("completions_2.json", current_batch, overwrite=False)
                    current_batch = []  # Reset after writing

    # Write any remaining items in the final batch
    if current_batch:
        write_to_json("completions_2.json", current_batch, overwrite=False)

    # After you're done with the model
    import torch
    torch.cuda.empty_cache()

