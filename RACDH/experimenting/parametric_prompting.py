from NER import NER
from data_utils import load_samples, generate_completion
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import StoppingCriteria, StoppingCriteriaList
import re
import torch


model_name_or_path = "meta-llama/Llama-3.1-8B-Instruct"
device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
model = AutoModelForCausalLM.from_pretrained(
    model_name_or_path,
    torch_dtype=torch.bfloat16
).to(device)


if __name__ == "__main__":
    sample_file = "MIND/auto-labeled/wiki/wiki_test.json"
    samples = load_samples(sample_file, 10)

    for i, sample in enumerate(samples):
        text = " ".join(sample['sentences'])
        user_prompt = f"""You are tasked to extract the most relevant entity that is not key to the text but is well-known and correlated with the topic. Please focus on proper nouns, such as names of people, places, or organizations. Return it as a single entity.
Example:
Text:
<<< Albert Einstein was a theoretical physicist who developed the theory of relativity. He was born in Ulm, Germany. >>>
Entity:
<<< Germany >>>

Another example:
Text:
<<< The Great Wall of China is a series of fortifications made of various materials, built to protect the northern borders of the Chinese Empire. It stretches across several provinces in northern China, including Hebei and Shanxi. >>>
Entity:
<<< Hebei >>>

Notice how both texts:
- Returns just the entity that is considered very well known to be tied to the subject, e.g., Einstein is born in Germany.
- The entity is captured using <<< and >>>

Now it is your turn. 
Text:
<<< {text} >>>
Entity:
<<<"""
        completion = generate_completion(model, tokenizer, device, user_prompt, max_new_tokens=20, temperature=0.5)

        pattern = re.compile(
            r'Entity:\s*'                     # The header line
            r'<<<\s*(.*?)\s*>>>'               # Capture everything between <<< and >>>
            r'(?:\})?\s*',                     # Optionally match a '}' and some whitespace
            flags=re.DOTALL
        )
        matches = re.findall(pattern, completion)
        entity = [entity.strip() for entity in matches][-1] if matches else []
        print("-" * 20 + sample["title"] + "-" * 20)
        print("-" * 5 + "Original text" + "-" * 5)
        print(text)
        print("-" * 5 + "Entity selected by LLM" + "-" * 5)
        print(entity)


        entity_ex = "Ulm"
        original_ex = "Albert Einstein was a theoretical physicist who developed the theory of relativity. He was born in Ulm, Germany."
        rewritten_ex = "Albert Einstein was a theoretical physicist who developed the theory of relativity. He was born in a town in Germany."
        user_prompt = f"""You are tasked to remove a given entity from a passage.


Entity:
<<< {entity_ex} >>>
Example passage:
<<< {original_ex} >>>
Rewritten passage:
<<< {rewritten_ex} >>>
Notice how the text:
- Maintains a similar sentence structure
- Only omits the entity and keeps all other information
- It still pertains the entity but doesn't explicitly name it.
- The sentence still reads natural, a person won't notice an entity is missing.

Now it is your turn

Below is a real Wikipedia passage. Transform it in the same style:

Entity:
<<< {entity} >>>>
Orginal Passage:
<<< {text} >>>
Rewritten passage (ending with >>>):
<<<"""
        completion = generate_completion(model, tokenizer, device, user_prompt, max_new_tokens=500, temperature=0.5)

        pattern = re.compile(
        r'Rewritten passage \(ending with >>>\):\s*'  # The header line
        r'<<<(.*?)>>>'                                       # Capture everything between <<< and >>>
        r'(?:\})?\s*',                                       # Optionally match a '}' and some whitespace
        flags=re.DOTALL
        )

        matches = re.findall(pattern, completion)
        # Clean up extra whitespace:
        cleaned_passages = [m.strip() for m in matches]

        print("-" * 5 + "Entity removed from passage" + "-" * 5)
        print(cleaned_passages)


        original_ex = rewritten_ex
        rewritten_ex = "Albert Einstein was a theoretical physicist who developed the theory of relativity. He was born in a town in Germany. This town was named Ulm."
        user_prompt2 = f"""You are tasked to introduce a named entity in a passage where it is not specifically named.
Entity:
<<< {entity} >>>
Example passage:
<<< {original_ex} >>>
Rewritten passage:
<<< {rewritten_ex} >>>
Notice how the text:
- Maintains a similar sentence structure.
- Introduces the entity by using its name.
- It is VERY important that the entity is mentioned clearly in the last sentence.
- The added sentence does not start with the name of the entity, but it introduces it naturally.
- The addition feel organic.

Now it is your turn

Below is a real Wikipedia passage. Transform it in the same style:

Entity:
<<< {entity} >>>>
Orginal Passage:
<<< {cleaned_passages[0]} >>>
Rewritten passage (ending with >>>):
<<< {cleaned_passages[0]} """
        
        completion = generate_completion(model, tokenizer, device, user_prompt2, max_new_tokens=500, temperature=0.5)
        pattern = re.compile(
        r'Rewritten passage \(ending with >>>\):\s*'  # The header line
        r'<<<(.*?)>>>'                                       # Capture everything between <<< and >>>
        r'(?:\})?\s*',                                       # Optionally match a '}' and some whitespace
        flags=re.DOTALL
        )

        matches = re.findall(pattern, completion)
        # Clean up extra whitespace:
        cleaned_passages = [m.strip() for m in matches]
        print("-" * 5 + "Entity reintroduced from passage" + "-" * 5)
        print(cleaned_passages)

        
