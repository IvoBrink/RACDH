import sys
import os
from tqdm import tqdm
sys.path.append(os.path.abspath("/home/ibrink/RACDH/RACDH/"))
from RACDH.data_generation.utils.reading_data import load_json
from RACDH.data_generation.utils.print import *
from RACDH.config import params
from RACDH.data_generation.removing.prompt_remove_entity import remove_entity_from_passage
from RACDH.data_generation.utils.writing_data import write_to_json



if __name__ == "__main__":
    prev_knowledge_target = "Llama-3.1-8B"
    prev_knowledge_instruct = "gpt-4o"
    samples = load_json(f"{prev_knowledge_target}/{prev_knowledge_instruct}/knowledge.json")

    data_to_save = []
    total, faulty = 0, 0
    for sample in tqdm(samples, desc="Processing samples"):
        if params.debug: print_h1(f"Rewrite passage [{sample['title']}]")
        title = sample["title"]
        passage = sample["passage"]
        known_entities = sample["known_entites"]
        if len(known_entities) == 0:
            continue


        rewritten_passages = {}
        for entity in known_entities.keys():
            total += 1
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

            if len(data_to_save) % 10 == 0:
                write_to_json("rewritten_known.json", data_to_save)

    write_to_json("rewritten_known.json", data_to_save)
    print(f"Faulty percentage: {(faulty/total)*100:.2f}%")

         # After you're done with the model
    import torch
    torch.cuda.empty_cache()

    


        