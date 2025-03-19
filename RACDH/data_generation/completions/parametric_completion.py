from RACDH.data_generation.instruct_model import generate_completion
from RACDH.data_generation.utils.print import *
from RACDH.config import params

def add_parametric_completion(entity, passage):
    prompt, pattern = get_prompt(entity, passage)
    completion = generate_completion(prompt, pattern, max_new_tokens=516, temperature=0.5, debug=params.debug)

    if entity.lower() not in completion.lower():
        print_warning("Entity not found in model output for parametric. Returning None.")
        return None

    sanitized_completion = remove_entity(completion, entity)
    
    final_passage = passage + " " + sanitized_completion
    return final_passage

def remove_entity(output, entity):
    lines = output.splitlines()
    truncated_output = []
    
    for line in lines:
        # Check if the entity is mentioned in the line (case-insensitive)
        if entity.lower() in line.lower():
            pos = line.lower().find(entity.lower())
            truncated_line = line[:pos]
            truncated_output.append(truncated_line.strip())
            break
        else:
            truncated_output.append(line)
    
    result = "\n".join(truncated_output)
    if params.debug:
        print_h4("Truncate entity")
        print(result)
    return result

def get_prompt(entity, passage):
    passage_ex = """Frankenstein is a gothic novel that was first published in 1818. The story follows a young scientist who creates a sapient creature through an unorthodox experiment, and it is often hailed as the first true work of science fiction."""
    entity_ex = "Mary ShelleyMary Shelley"
    output_ex = "The author of the novel was named Mary Shelley."
    prompt = f"""Instruction: You are given a passage describing something but omitting an entity's name. Then you are given that entity’s name. Produce one additional sentence that naturally appends to the passage and introduces the entity by relating it to the passage. You may use one of the following formats (or something similarly natural):
- “This was called [entity].”
- “It was named [entity].”
- “The [description] was called [entity].”
- “Locals called it [entity].”
- “They referred to her as [entity].”

IMPORTANT:
1. You must use the exact entity name as provided—no alterations, changes in capitalization, or partial usage.
2. Provide only that single sentence to append.

One-Shot Example

Passage:
<<< {passage_ex} >>>

Entity:
<<< {entity_ex} >>>

Output:
<<< {output_ex} >>>

Notice how:
- The entity is introduced precisely as given.
- Only one additional sentence is added to the end of the passage.

Now it is your turn.

Below is a real Wikipedia passage and an entity. Please provide your single-sentence output in the same format:

Entity:
<<< {entity} >>>

Original Passage:
<<< {passage} >>>

Output (ending with >>>):
<<<"""
    pattern = r'Output \(ending with >>>\):\s*<<<(.*?)>>>\s*(?:\})?'
    return prompt, pattern
