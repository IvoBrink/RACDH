import sys
import os
import json
from tqdm import tqdm
sys.path.append(os.path.abspath("/home/ibrink/RACDH/RACDH/"))
from RACDH.data_generation.utils.reading_data import load_samples_wiki, load_samples_new_wiki
from RACDH.data_generation.entity_recognition.spaCy import get_entities
from RACDH.data_generation.utils.print import highlight_entities
from RACDH.data_generation.entity_recognition.selection import select_best_entities
from RACDH.data_generation.utils.writing_data import write_to_json
from RACDH.data_generation.utils.normalize_punctuation import normalize_punctuation
import re


#TODO: not include entities that are part of a summation, makes QA and completions hard!
if __name__ == "__main__":
    n_samples = 10
    # samples = load_samples_wiki("wiki_train.json", n_samples, check_earlier_created_data=True)
    samples = load_samples_new_wiki("random_wiki_articles.ndjson", n_samples)
    total_length = 0
    data_to_save = []

    for sample in tqdm(samples, desc="Processing samples"):
        # text = " ".join(sample["sentences"])
        text = sample["text"]
        # text = normalize_punctuation(text)
        title = sample["title"]

        entities = get_entities(text)
        entities = select_best_entities(text, entities, title)
        correct_entities = [entity for entity, _, status in entities if status == "Correct"]
        correct_entities = list(set(correct_entities))

        print("-" * 10 + sample["title"] + "-" * 10)
        print(highlight_entities(text, entities))
        n_correct_entities = len(correct_entities)
        total_length += n_correct_entities
        print("-"*10 + str(n_correct_entities) + "-" * 10)

        if n_correct_entities > 0:
            data_to_save.append({
                "title": title,
                "passage": text,
                "entities": correct_entities
            })

        if len(data_to_save) % 100 == 0:
            write_to_json("extracted_entities_2.json", data_to_save, ignore_dirs=True, overwrite=False)
            
    print(f"Average number of entitites {total_length/len(samples)}")
    print(f"Total entities: {total_length}")
    print(f"Total articles processed: {len(samples)}")

    write_to_json("extracted_entities_2.json", data_to_save, ignore_dirs=True, overwrite=False)

         # After you're done with the model
    from RACDH.data_generation.target_model import taget_model
    from RACDH.data_generation.instruct_model import instruct_model
    import torch
    del instruct_model
    del taget_model
    torch.cuda.empty_cache()
    
    