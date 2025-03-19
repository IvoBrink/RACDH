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


if __name__ == "__main__":
    samples_known = load_json(f"{params.target_name}/gpt-4o-mini/rewritten_known.json")
    samples = load_json(f"{params.target_name}/gpt-4o/knowledge.json")
    data_to_save = []

    # Parametric
    if params.debug: print_h1("Generating Parametric completion examples")
    for sample_known in tqdm(samples_known, desc="Processing Parametric samples"):
        title = sample_known["title"]
        passage = sample_known["passage"]
        rewritten_passages = sample_known["rewritten_passages"]
        if len(rewritten_passages) == 0:
            continue
        if params.debug: print_h2(f"Passage [{title}]")

        for entity, rewritten_passage in rewritten_passages.items():
            if params.debug: print_h3(f"Entity [{entity}]")
            final_passage = add_parametric_completion(entity, rewritten_passage)
            if final_passage is not None:
                data_to_save.append({
                    "title" : title,
                    "passage" : final_passage,
                    "entity" : entity,
                    "label" : "parametric",
                    "original_passage" : passage
                })
                if len(data_to_save) % 10 == 0:
                    write_to_json("completions.json", data_to_save)


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
            final_passage = add_contextual_completion(entity, passage)
            if final_passage is not None:
                data_to_save.append({
                    "title" : title,
                    "passage" : final_passage,
                    "entity" : entity,
                    "label" : "contextual",
                    "original_passage" : passage
                })
                if len(data_to_save) % 10 == 0:
                    write_to_json("completions.json", data_to_save)
    

    # Shuffle the data before final save
    random.shuffle(data_to_save)
    write_to_json("completions.json", data_to_save)
    #  After you're done with the model
    import torch
    torch.cuda.empty_cache()

