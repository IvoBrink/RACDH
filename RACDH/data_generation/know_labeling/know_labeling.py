import sys, os, torch
from collections import defaultdict      ### NEW
from tqdm import tqdm

sys.path.append(os.path.abspath("/home/ibrink/RACDH/RACDH/"))
from RACDH.data_generation.know_labeling.design_completions.all_completion_types import generate_all_knowledge_tests
from RACDH.data_generation.know_labeling.generate_completions.target_completion_knowing import evaluate_knowledge
from RACDH.data_generation.utils.reading_data import load_json
from RACDH.data_generation.utils.writing_data import write_to_json
from RACDH.data_generation.utils.print import *
from RACDH.config import params


def deduplicate_questions(raw):
    """Return *dict* keyed (title, entity) ➜ question_dict (keeps first seen)."""
    dedup = {}
    for q in raw:
        key = (q["title"], q["entity"])
        dedup.setdefault(key, q)
    return dedup


if __name__ == "__main__":

    # ------------ 1. Load once, build fast-lookup structures -------------
    samples = load_json("extracted_entities_2.json")

    # Existing Q&A – keep both dict (for O(1) look-ups) and list (for writing)
    existing_q_map = deduplicate_questions(load_json("knowledge_questions.json"))  ### CHANGED
    existing_q_keys = set(existing_q_map)                                          ### CHANGED

    existing_samples = load_json(f"{params.target_name}/{params.instruct_name}/knowledge_2.json")
    existing_titles = {s["title"] for s in existing_samples}                       ### CHANGED

    # ------------ 2. Book-keeping counters --------------------------------
    succesful_tests_generated = 0
    n_entities = sum(len(s["entities"]) for s in samples)
    l_known = l_ignored = 0

    current_batch = []

    # ------------ 3. Main loop --------------------------------------------
    for sample in tqdm(samples, desc="Processing samples"):
        title, passage, entities = sample["title"], sample["passage"], sample["entities"]

        if title in existing_titles:      # already processed for THIS target model
            if params.debug: print_h2("Data already exists for this target model!")
            continue

        known_ents, ignored_ents = {}, []

        for entity in entities:
            key = (title, entity)

            # ---------- 3a. Re-use existing question data if present ------
            q_data = existing_q_map.get(key)                                     ### CHANGED
            if q_data:
                tests = [v for v in (
                    q_data.get("alice_bob_conversation"),
                    q_data.get("truncated_passage"),
                    q_data.get("question")
                ) if v]                                                          # drop Nones
            else:
                tests = generate_all_knowledge_tests(passage, entity)
                existing_q_map[key] = {                                          ### CHANGED
                    "title": title,
                    "entity": entity,
                    "alice_bob_conversation": tests[0] if len(tests) > 0 else None,
                    "truncated_passage"    : tests[1] if len(tests) > 1 else None,
                    "question"             : tests[2] if len(tests) > 2 else None
                }

            succesful_tests_generated += len(tests)

            # ---------- 3b. Evaluate knowledge ----------------------------
            tests_passed = evaluate_knowledge(tests, entity)
            if tests_passed >= params.knowledge_tests_threshold:
                known_ents[entity] = tests_passed
            else:
                ignored_ents.append(entity)

        # ---------- 3c. Persist periodically ------------------------------
        current_batch.append({
            "title": title,
            "passage": passage,
            "entities": entities,
            "known_entites": known_ents,
            "ignored_entities": ignored_ents
        })
        l_known   += len(known_ents)
        l_ignored += len(ignored_ents)

        if len(current_batch) > 10:
            write_to_json("knowledge_2.json", current_batch, ignore_dirs=False, overwrite=False)
            write_to_json("knowledge_questions.json", list(existing_q_map.values()),  ### CHANGED
                          ignore_dirs=True, overwrite=True)
            current_batch = []

    # ------------ 4. Final flush & cleanup --------------------------------
    if current_batch:
        write_to_json("knowledge_2.json", current_batch, ignore_dirs=False, overwrite=False)

    write_to_json("knowledge_questions.json", list(existing_q_map.values()),       ### CHANGED
                  ignore_dirs=True, overwrite=True)

    torch.cuda.empty_cache()
