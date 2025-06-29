import sys
import os
from tqdm import tqdm
sys.path.append(os.path.abspath("/home/ibrink/RACDH/RACDH/"))
from RACDH.data_generation.utils.reading_data import load_json
from RACDH.data_generation.utils.print import *
from RACDH.config import params
from RACDH.data_generation.removing.prompt_remove_entity import remove_entity_from_passage
from RACDH.data_generation.utils.writing_data import write_to_json



def map_to_dict_passage(samples):
    existing_dict = {}
    for sample in samples:
        title = sample["title"]
        for entity, rewritten in sample["rewritten_passages"].items():
            existing_dict[(title, entity)] = rewritten
    return existing_dict





if __name__ == "__main__":
    samples = load_json(f"{params.target_name}/{params.instruct_name}/knowledge_2.json")
    samples_existing = load_json(f"{params.target_name}/{params.instruct_name}/rewritten_known_2.json")
    existing_titles = [s["title"] for s in samples_existing]


    samples_different_target = load_json(f"Llama-3.1-8B/{params.instruct_name}/rewritten_known_2.json") 
    dict_different_target = map_to_dict_passage(samples_different_target)

    data_to_save = []
    total, faulty = 0, 0
    for sample in tqdm(samples, desc="Processing samples"):
        if params.debug: 
            print_h1(f"Rewrite passage [{sample['title']}]")
            print(sample["passage"])
        title = sample["title"]
        passage = sample["passage"]
        known_entities = sample["known_entites"]
        if len(known_entities) == 0 or title in existing_titles:
            continue


        rewritten_passages = {}
        for entity in known_entities.keys():
            key = (title, entity)
            total += 1
            rewritten_data = dict_different_target.get(key)
            if rewritten_data:
                print_h3(f"Rewritten paragraph found for {key}")
                rewritten_passages[entity] = rewritten_data
            else:
                rewritten_passage = remove_entity_from_passage(entity, passage)
                if rewritten_passage is None:
                    faulty += 1
                    continue
                rewritten_passages[entity] = rewritten_passage
        
        if len(rewritten_passages) > 0:
            data_to_save.append({
                "title":title,
                "passage": passage,
                "rewritten_passages" : rewritten_passages,
                "known_entites" : known_entities
            })

            if len(data_to_save) > 10:
                write_to_json("rewritten_known_2.json", data_to_save, overwrite=False)
                data_to_save = []
    if data_to_save:
        write_to_json("rewritten_known_2.json", data_to_save, overwrite=False)
    print(f"Faulty percentage: {(faulty/total)*100:.2f}%")

    # After you're done with the model
    import torch
    torch.cuda.empty_cache()

    


        