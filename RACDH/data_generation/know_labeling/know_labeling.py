import sys
import os
sys.path.append(os.path.abspath("/home/ibrink/RACDH/RACDH/"))
from RACDH.data_generation.know_labeling.design_completions.all_completion_types import generate_all_knowledge_tests
from RACDH.data_generation.know_labeling.generate_completions.target_completion_knowing import evaluate_knowledge
from RACDH.data_generation.utils.reading_data import load_sample_entities
from RACDH.data_generation.utils.writing_data import write_to_json
from RACDH.config import params
from tqdm import tqdm
from RACDH.data_generation.utils.print import *


if __name__ == "__main__":
    samples = load_sample_entities("extracted_entities.json")

    succesful_tests_generated = 0
    n_entities = sum(len(sample["entities"]) for sample in samples)

    knowledge_dict = {}
    data_to_save = []

    for sample in tqdm(samples, desc="Processing samples"):
        passage = sample["passage"]
        entities = sample["entities"]
        title = sample["title"]
        if params.debug: print_h1(title)
        
        known_ents = []
        ignored_ents = []
        for entity in entities:
            if params.debug: print_h2(f"Generating know-tests for {entity}")
            tests = generate_all_knowledge_tests(passage, entity)
            succesful_tests_generated += len(tests)

            if params.debug: print_h2(f"Completing tasks for {entity}")

            if evaluate_knowledge(tests, entity):
                known_ents.append(entity)
            else:
                ignored_ents.append(entity)

        data_to_save.append({
            "title" : title,
            "passage" : passage,
            "entities" : entities,
            "known_entites" : known_ents,
            "ignored_entities" : ignored_ents
        })

            
            
    write_to_json(f"{params.taget_model_name_or_path.split('/')[-1]}_knowledge.json", data_to_save)
    print(f"Number of sucesful completions per entity: {succesful_tests_generated/n_entities}")