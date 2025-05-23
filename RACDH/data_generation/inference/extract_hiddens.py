import sys
import os
import random
from tqdm import tqdm
sys.path.append(os.path.abspath("/home/ibrink/RACDH/RACDH/"))
from RACDH.data_generation.utils.reading_data import load_json
from RACDH.config import params
from RACDH.data_generation.target_model import generate_completion_extract_hiddens
from RACDH.data_generation.utils.print import *
from RACDH.data_generation.utils.writing_data import *
from RACDH.data_generation.inference.entity_tokens_find import get_entity_span_text_align, reconstruct_generated_text
from collections import defaultdict
from RACDH.data_generation.instruct_model import generate_completion_GPT
    


if __name__ == "__main__":
    samples = load_json(f"{params.target_name}/{params.instruct_name}/completions_rewrite.json")
    random.shuffle(samples) # Contextual and Parametric are in order

    success = defaultdict(int)
    total = defaultdict(int)
    
    # This list holds actual hidden states for *only* successful matches
    all_hidden_states = []
    
    # This list will hold sample metadata for only successful matches
    data_to_save = []
    
    # We'll track a separate index that increments only when we have a successful match
    hidden_index = 0

    for sample in tqdm(samples, desc="Processing samples"):
        title = sample["title"]
        completion = sample["passage"]
        entity = sample["entity"]
        label = sample["label"]
        
        print_h1(f"Title [{title}]")
        print_h2(label)
        print_h3(entity)
        print(completion)
        
        # Tally up total for success-rate printing later
        total[label] += 1

        # Generate text + hidden states
        token_info = generate_completion_extract_hiddens(
            prompt=completion,
            max_new_tokens=10,
            temperature=0.5,
            debug=False
        )
        
        # Attempt to locate entity within token_info
        entity_info = get_entity_span_text_align(token_info, entity)

        if not entity_info:
            # If no entity match found, skip saving entirely
            print_h2(f"No match found for entity: {entity}")
            continue
        else:
            # We do have a match
            success[label] += 1

            matched_tokens = entity_info["tokens"]  # List of dicts with "step", "token_str", etc.
            print_h2(f"Match found for entity: {entity}")
            print("Entity tokens in the matched span:")
            for token_data in matched_tokens:
                tok_str = token_data["token_str"]
                step_idx = token_data["step"]
                print(f"  Step {step_idx}: {repr(tok_str)}")

            # Append to big list of hidden states
            all_hidden_states.append({
                "first_token_hidden": entity_info["first_token_hidden"],
                "last_token_hidden": entity_info["last_token_hidden"],
            })
            
            # Add the metadata, referencing hidden_index
            data_to_save.append({
                "title": title,
                "passage": completion,
                "entity": entity,
                "generated": reconstruct_generated_text(token_info),
                "label": label,
                "hidden_states_index": hidden_index,
                "similar_entity" : entity_info["similar_entity"] 
            })

            # Only increment hidden_index for successful matches
            hidden_index += 1

    # Save the hidden states to a single .pt file
    write_tensors("hiddens_rewrite.pt", all_hidden_states)

    # Save the metadata to JSON
    write_to_json("hiddens_metadata_rewrite.json", data_to_save)

    # Print success rates
    # (Handle the case if there are zero 'contextual' or 'parametric' to avoid division by zero)
    if total["contextual"] > 0:
        print(f"Success rate CONTEXTUAL: {success['contextual'] / total['contextual']:.2%}")
    else:
        print("No contextual samples.")

    if total["parametric"] > 0:
        print(f"Success rate PARAMETRIC: {success['parametric'] / total['parametric']:.2%}")
    else:
        print("No parametric samples.")

    # Overall success
    if len(samples) > 0:
        overall_success = (success['parametric'] + success['contextual']) / len(samples)
        print(f"Success rate OVERALL: {overall_success:.2%}")
    else:
        print("No samples found.")
