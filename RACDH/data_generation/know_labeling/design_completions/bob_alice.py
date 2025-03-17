from RACDH.data_generation.instruct_model import generate_completion
from RACDH.data_generation.utils.print import *
from RACDH.config import params
import re


def generate_alice_bob_example(passage, entity):
    if params.debug: print_h3("Alice-bob generation")
    prompt, pattern = get_prompt(passage, entity)
    completion = generate_completion(prompt, pattern, max_new_tokens=256, temperature=0.5, debug=params.debug)
    completion = completion.replace(" >>>", "")
    if sanity_checks(completion, entity):
        output = remove_entity(completion, entity)
        return output
    else:
        return None


def sanity_checks(completion: str, entity: str):
    """
    Checks that:
    1) The entity is not mentioned before Bob speaks.
    2) Bob explicitly names the entity in at least one of his lines.
    
    Raises ValueError if these conditions are not met.
    Returns True otherwise.
    """

#   TODO: also check if very similar or synonym of entity is mentioned by alice -> also invalid

    lines = completion.splitlines()
    
    seen_bob_line = False
    found_entity_in_bob = False
    found_entity_before_bob = False

    for line in lines:
        # Normalize leading/trailing whitespace
        line_stripped = line.strip()

        # Check if this line starts with "Bob:" (case-insensitive)
        if re.match(r'^bob:', line_stripped, re.IGNORECASE):
            seen_bob_line = True
            # If the entity is in Bob's line, mark it
            if entity in line_stripped:
                found_entity_in_bob = True
        else:
            # If we haven't seen Bob yet and the entity shows up in Alice/other lines, that's invalid
            if not seen_bob_line and (entity in line_stripped):
                found_entity_before_bob = True

    if found_entity_before_bob:
        if params.debug: print_warning("Bob-Alice: The entity was mentioned before Bob spoke. Omitting example.")
        return False
    if not found_entity_in_bob:
        if params.debug: print_warning("Bob-Alice: Bob never mentioned the entity by name. Omitting example.")
        return False

    return True
    

def remove_entity(output, entity):
    # Split the output into lines
    lines = output.splitlines()
    truncated_output = []

    for line in lines:
        # Check if the entity is mentioned in the line
        if entity in line:
            # If the entity is found, truncate the line at the entity's mention
            truncated_line = line.split(entity)[0]
            truncated_output.append(truncated_line.strip())
            break
        else:
            # If the entity is not found, keep the line
            truncated_output.append(line)
    result = "\n".join(truncated_output)
    if params.debug:
        print_h4("Truncate entity")
        print(result)
    return result

def get_prompt(passage, entity):
    oneshot_passage = """The Battle of Evesham ( 4 August 1265 ) was one of the two main battles of 13th century England 's Second Barons ' War . It marked the defeat of Simon de Montfort , Earl of Leicester , and the rebellious barons by Prince Edward – later King Edward I – who led the forces of his father , King Henry III . It took place on 4 August 1265 , near the town of Evesham , Worcestershire ."""
    oneshot_entity = """King Henry III"""
    oneshot_output = """Alice: I can't remember exactly who was the king of England in 1265 during the Battle of Evesham . I can't remember.
Bob: Actually, I know. It was King Henry III."""

    prompt = f"""You will receive a Wikipedia passage of an arbitrary topic and an entity that is mentioned somewhere within the passage. You will create a dialogue between Alice and Bob. Alice can't think of the name of [entity]. She describes it perfectly using the Wikipedia passage. Bob is all-knowing, and tells Alice the name of the entity. Alice is not allowed to say the name of the entity or part of the entity, and must ONLY give information provided in the Wikipedia passage to describe said entity. Bob MUST say the exact name of the entity. Here is an example.
Wikipedia passage:
<<< {oneshot_passage}  >>>
Entity:
<<< {oneshot_entity} >>>
Alice-bob conversation:
<<< {oneshot_output} >>>

Now it is your turn:
Wikipedia passage:
<<< {passage} >>>
Entity:
<<< {entity} >>>
Alice-bob conversation (ending with >>>):
<<< """

    pattern = r'Alice-bob conversation \(ending with >>>\):\s*<<<(.*?)>>>\s*(?:\})?'
    return prompt, pattern