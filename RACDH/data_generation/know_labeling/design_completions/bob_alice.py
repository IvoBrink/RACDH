from RACDH.data_generation.instruct_model import generate_completion
import re


def generate_alice_bob_example(passage, entity):
    prompt, pattern = get_prompt(passage, entity)
    completion = generate_completion(prompt, pattern, max_new_tokens=256, temperature=0.5, debug=True)
    # TODO: sanity checks here, e.g., is the entity named by bob (and NOT alice)
    output = remove_entity(completion, entity)
    return output

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

    return "\n".join(truncated_output)

def get_prompt(passage, entity):
    oneshot_passage = """The Battle of Evesham ( 4 August 1265 ) was one of the two main battles of 13th century England 's Second Barons ' War . It marked the defeat of Simon de Montfort , Earl of Leicester , and the rebellious barons by Prince Edward – later King Edward I – who led the forces of his father , King Henry III . It took place on 4 August 1265 , near the town of Evesham , Worcestershire ."""
    oneshot_entity = """King Henry III"""
    oneshot_output = """Alice: I can't remember exactly who was the king of England in 1265 during the Battle of Evesham . I can't remember.
Bob: Actually, I know. It was King Henry III."""

    prompt = f"""You will receive a Wikipedia passage of an arbitrary topic and an entity that is mentioned somewhere within the text. Like so:
Wikipedia passage:
<<< {oneshot_passage}  >>>
Entity:
<<< {oneshot_entity} >>>
Alice-bob conversation:
<<< {oneshot_output} >>>

Notice how a dialogue between Alice and Bob was created regarding the entity and its relatedness to the passage. Make sure that the entity is referred to by its name!
Alice is not allowed to say the name of the entity, and can ONLY give information provided in the Wikipedia passage. Bob MUST say the name of the entity.

Now it is your turn:
Wikipedia passage:
<<< {passage} >>>
Entity:
<<< {entity} >>>
Alice-bob conversation (ending with >>>):
<<< """
    with open("generated_prompt.txt", "a") as file:  # Change to append mode
        file.write(prompt + "\n")  # Add a newline for separation

    pattern = r'Alice-bob conversation \(ending with >>>\):\s*<<<(.*?)>>>\s*(?:\})?'
    return prompt, pattern