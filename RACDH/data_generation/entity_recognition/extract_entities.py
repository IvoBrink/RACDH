import sys
import os
import json
sys.path.append(os.path.abspath("/home/ibrink/RACDH/RACDH/"))
from RACDH.data_generation.utils.reading_data import load_samples_wiki
from RACDH.data_generation.entity_recognition.spaCy import get_entities
from RACDH.data_generation.utils.print import highlight_entities
from RACDH.data_generation.entity_recognition.selection import select_best_entities
from RACDH.data_generation.utils.writing_data import write_to_json



if __name__ == "__main__":
    n_samples = 30
    samples = load_samples_wiki("wiki_train.json", n_samples)
    total_length = 0
    data_to_save = []
    for sample in samples:
        text = " ".join(sample['sentences'])
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

        data_to_save.append({
            "title": title,
            "passage": text,
            "entities": correct_entities
        })
    
    print(f"Average number of entitites {total_length/n_samples}")

    write_to_json("extracted_entities.json", data_to_save)

    # Save the data to a JSON file
    
    