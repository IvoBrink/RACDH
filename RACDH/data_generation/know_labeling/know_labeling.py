import sys
import os
sys.path.append(os.path.abspath("/home/ibrink/RACDH/RACDH/"))
from RACDH.data_generation.know_labeling.design_completions.all_completion_types import generate_all_knowledge_tests
from RACDH.data_generation.know_labeling.generate_completions.target_completion_knowing import evaluate_knowledge
from RACDH.data_generation.utils.reading_data import load_json
from RACDH.data_generation.utils.writing_data import write_to_json
from RACDH.config import params
from tqdm import tqdm
from RACDH.data_generation.utils.print import *


if __name__ == "__main__":

    samples = load_json("extracted_entities.json")
    existing_questions = load_json("knowledge_questions.json")
    
    # Remove any duplicates based on title and entity
    seen = set()
    unique_questions = []
    for q in existing_questions:
        key = (q["title"], q["entity"])
        if key not in seen:
            seen.add(key)
            unique_questions.append(q)
    existing_questions = unique_questions
    
    existing_questions_titles = [s["title"] for s in existing_questions]

    existing_samples = load_json(f"{params.target_name}/{params.instruct_name}/knowledge.json")
    existing_titles = [s["title"] for s in existing_samples]

    succesful_tests_generated = 0
    n_entities = sum(len(sample["entities"]) for sample in samples)

    knowledge_dict = {}
    data_to_save = []
    l_known, l_ignored = 0,0

    for sample in tqdm(samples, desc="Processing samples"):
        passage = sample["passage"]
        entities = sample["entities"]
        title = sample["title"]

        if params.debug: print_h1(title)
        
        # Skip processing if title exists and use existing data for this target model
        if title in existing_titles:
            original_data = next((item for item in existing_samples if item["title"] == title), None)
            if original_data:
                original_data["passage"] = passage
                data_to_save.append(original_data)
                if params.debug: print_h2("Data already exists for this target model!")
                continue


        known_ents = {}
        ignored_ents = []
        for entity in entities:
            if params.debug: print_h2(f"Generating know-tests for {entity}")

            question_data = next((item for item in existing_questions if item["title"] == title and item["entity"] == entity), None)
            if question_data:
                if params.debug:
                    print(title, entity)
                    print("Data found for title and entity")
                # Extract existing tests, ensuring we don't have None values
                tests = []
                if question_data.get("alice_bob_conversation"):
                    tests.append(question_data["alice_bob_conversation"])
                if question_data.get("truncated_passage"):
                    tests.append(question_data["truncated_passage"])
                if question_data.get("question"):
                    tests.append(question_data["question"])
            else:
                tests = generate_all_knowledge_tests(passage, entity)
                existing_questions.append({
                    "title": title,
                    "entity": entity,
                    "alice_bob_conversation": tests[0] if len(tests) > 0 else None,
                    "truncated_passage": tests[1] if len(tests) > 1 else None,
                    "question": tests[2] if len(tests) > 2 else None
                })
            succesful_tests_generated += len(tests)

            if params.debug: print_h2(f"Completing tasks for {entity}")

            tests_passed = evaluate_knowledge(tests, entity)
            if  tests_passed >= params.knowledge_tests_threshold:
                known_ents[entity] = tests_passed
            else:
                ignored_ents.append(entity)

        data_to_save.append({
            "title": title,
            "passage": passage,
            "entities": entities,
            "known_entites": known_ents,
            "ignored_entities": ignored_ents
        })

        l_known += len(known_ents)
        l_ignored += len(ignored_ents)

        if len(data_to_save) % 10 == 0:
            write_to_json("knowledge.json", data_to_save, ignore_dirs=False, overwrite=False)
            write_to_json("knowledge_questions.json", existing_questions, ignore_dirs=True, overwrite=True)

    write_to_json("knowledge.json", data_to_save, ignore_dirs=False, overwrite=False)
            
    # print(f"Number of sucesful completions per entity: {succesful_tests_generated/n_entities}")
    # print(f"{round((l_known/n_entities)*100,2)}% are known entities")
    # print(f"{round((l_ignored/n_entities)*100,2)}% are ignored entities")

    # Write the updated questions to file
    write_to_json("knowledge_questions.json", existing_questions, ignore_dirs=True, overwrite=True)

     # After you're done with the models
    import torch
    torch.cuda.empty_cache()