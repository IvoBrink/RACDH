from RACDH.data_generation.instruct_model import generate_completion
from RACDH.data_generation.utils.print import *
from RACDH.config import params
import re


def question_example(passage, entity):
    if params.debug: print_h3("Question generation")
    prompt, pattern = get_prompt(passage, entity)
    completion = generate_completion(prompt, pattern, max_new_tokens=256, temperature=0.5, debug=params.debug)
    if sanity_checks(completion, entity):
        output = remove_entity(completion, entity)
        return output
    else:
        return None


def sanity_checks(output, entity):
    lines = output.splitlines()
    for line in lines:
        if "A: " in line:
            answer = line[3:].strip()
            if entity.lower() in answer.lower():
                return True
    print_warning("Answer to question is not exactly the entity.")
    return False



def remove_entity(output, entity):
    # Split the output into lines
    lines = output.splitlines()
    truncated_output = []

    for line in lines:
        # Check if the entity is mentioned in the line (case-insensitive)
        if entity.lower() in line.lower():
            # If the entity is found, truncate the line at the entity's mention
            # Find the actual position considering case
            pos = line.lower().find(entity.lower())
            truncated_line = line[:pos]
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
    oneshot_output = """Q: Who was the king of England during the Battle of Evesham in the 13th centure?
A: King Henry III"""

    prompt = f"""You will receive a Wikipedia passage of an arbitrary topic and an entity that is mentioned somewhere within the text. Like so:
Wikipedia passage:
<<< {oneshot_passage}  >>>
Entity:
<<< {oneshot_entity} >>>
Alice-bob conversation:
<<< {oneshot_output} >>>

Follow the same pattern using a new Wikipedia passage. You generate a question (Q) to which the answer is the entity (A).

Now it is your turn:
Wikipedia passage:
<<< {passage} >>>
Entity:
<<< {entity} >>>
Question-answer (ending with >>>):
<<< """

    pattern = r'Question-answer \(ending with >>>\):\s*<<<(.*?)>>>\s*(?:\})?'
    return prompt, pattern