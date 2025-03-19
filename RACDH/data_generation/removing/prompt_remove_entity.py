from RACDH.data_generation.instruct_model import generate_completion
from RACDH.config import params
from RACDH.data_generation.utils.print import *

def remove_entity_from_passage(entity, passage):
    if params.debug: print_h2(f"Rewrite passage for [{entity}]")
    prompt, pattern = get_prompt(entity, passage)
    completion = generate_completion(prompt, pattern, max_new_tokens=516, temperature=0.5, debug=params.debug)
    if sanity_checks(entity, completion):
        return completion
    else:
        print_warning("Entity not succesfully removed from passage")
        return None


def sanity_checks(entity, completion):
    # Simple case-insensitive check to see if the entity is still present
    if entity.lower() in completion.lower():
        return False
    return True


def get_prompt(entity, passage):
    entity_ex = "United Kingdom"
    original_ex = """Winston Churchill was a British statesman, soldier, and writer who served as 
Prime Minister of the United Kingdom from 1940 to 1945 and again from 1951 to 1955. 
He led Britain to victory in the Second World War. Among the British public, he is 
widely considered the greatest Briton of all time. He was born to an aristocratic 
family in Oxfordshire, England."""
    
    rewritten_ex = """Winston Churchill was a statesman, soldier, and writer who served as Prime Minister from 1940 to 1945 and again from 1951 to 1955. He led people to victory 
in the Second World War. Among the public there, he is widely considered one of the 
greatest leaders of all time. He was born to an aristocratic family in Oxfordshire."""

    prompt = f"""
You are tasked with removing any given entity from a passage **and all similar or abbreviated references to it**. In other words, if the entity is "United States of America", you should remove or generalize "United States of America", 
"USA", "US", or any other obvious textual variations referring to that entity.
Entity:
<<< {entity_ex} >>>
Example passage:
<<< {original_ex} >>>
Rewritten passage:
<<< {rewritten_ex} >>>
Notice how the text:
- Maintains a similar sentence structure
- Omits the entity and **all** variant references (full form, abbreviations, synonyms)
- Keeps all other information intact
- Reads naturally so a casual reader does not notice an entity was removed

Now it is your turn

Below is a real Wikipedia passage. Transform it in the same style:

Entity:
<<< {entity} >>>
Original Passage:
<<< {passage} >>>
Rewritten passage (ending with >>>):
<<<
""".strip()

    # The regex pattern that captures the text inside Rewritten passage: <<< >>>
    pattern = r'Rewritten passage \(ending with >>>\):\s*<<<(.*?)>>>\s*(?:\})?'
    return prompt, pattern

