import sys
import os
sys.path.append(os.path.abspath("/home/ibrink/RACDH/RACDH/"))
from RACDH.data_generation.utils.reading_data import load_samples
from RACDH.data_generation.entity_recognition.spaCy import get_entities
from RACDH.data_generation.utils.print import highlight_entities
from RACDH.data_generation.entity_recognition.selection import select_best_entities



if __name__ == "__main__":
    n_samples = 100
    samples = load_samples("wiki_train.json", n_samples)
    total_length = 0
    for sample in samples:
        text = " ".join(sample['sentences'])
        title = sample["title"]
        entities = get_entities(text)
        entities = select_best_entities(text, entities, title)
        n_correct_entities = len([entity for entity, i, status in entities if status == "Correct"])
        total_length += n_correct_entities
        print("-" * 10 + sample["title"] + "-" * 10)
        print(highlight_entities(text, entities))
        print("-"*10 + str(n_correct_entities) + "-" * 10)
    print(f"Average number of entitites {total_length/n_samples}")
    