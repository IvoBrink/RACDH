import json
import random
import torch
import json
import re
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
from data_utils import load_samples, generate_completion
import os
from tqdm import tqdm

model_name_or_path = "meta-llama/Llama-3.1-8B-Instruct"
device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
model = AutoModelForCausalLM.from_pretrained(
    model_name_or_path,
    torch_dtype=torch.bfloat16
).to(device)


with open('RACDH/data/merged_unique_names.json', 'r') as f:
    PER_names = json.load(f)

with open('RACDH/data/nouns.json', 'r') as f:
    NOUNS = json.load(f)



def extract_rewritten_passages(text):
    """
    Finds all `Example rewritten passage (ending with >>>):` blocks
    and captures everything enclosed by <<<...>>>, optionally followed by '}'.

    Returns a list of the extracted passages (one per occurrence).
    """
    # Explanation:
    # 1) We match literal text: Example rewritten passage (ending with >>>):
    # 2) \s* = optional whitespace/newlines
    # 3) <<<(.*?)>>> = capture everything in triple angle brackets (DOTALL for multiline)
    # 4) (?:\})? = optionally match a single curly brace
    # 5) We do not require anything else after that. 
    pattern = re.compile(
        r'Example rewritten passage \(ending with >>>\):\s*'  # The header line
        r'<<<(.*?)>>>'                                       # Capture everything between <<< and >>>
        r'(?:\})?\s*',                                       # Optionally match a '}' and some whitespace
        flags=re.DOTALL
    )

    matches = re.findall(pattern, text)
    # Clean up extra whitespace:
    cleaned_passages = [m.strip() for m in matches]
    return cleaned_passages


def test_prompts(samples):
    # Provide a single example inside the prompt to show how to do the rewrite
    example_original = (
        "Original example passage:\n"
        "<<< Neil Armstrong was an American astronaut and the first person to walk on the Moon. He was born on August 5, 1930, in Wapakoneta, Ohio, and joined NASA in 1962. >>>"
    )
    example_rewritten = (
        "Example rewritten passage:\n"
        "<<< Stephanie Richmond was a renowned astronaut and the first person to step onto the Celestial Reef. She was born on August 3, 1925, in the city-state of Montera, and enlisted in the Radiant Fleet Academy in 1989. >>>"
    )

    for i, sample in tqdm(enumerate(samples, 1), total=len(samples), desc="Processing samples"):
        title = sample['title']
        sentences = " ".join(sample['sentences'])
        
        user_prompt = f"""{example_original}

{example_rewritten}

Notice how the new text:
- Maintains a similar sentence structure.
- Replaces all real names, dates (only the numbers, do NOT invent names for months), places, and organizations with entirely made-up ones. 
- Keeps an encyclopedic tone but no verifiable real facts remain.

Now It is Your Turn

Below is your **real** Wikipedia passage. Transform it in the same style: 
1. Keep the paragraph count and approximate sentence structure.  
2. Replace names, locations, dates, and other facts with imaginary ones. Use American-sounding names!
3. Output only your final fictional rewrite.  
4. Do not add extra commentary or disclaimers.
5. End your output after your last fictional sentence
Put >>> at the end to indicate that the passage has been finished

You must use these five names when replacing persons: {', '.join(random.sample(PER_names, 5))}!


Original passage:
<<< {sentences} >>>
Example rewritten passage (ending with >>>):
<<< """
        #You must use these five nouns for name inspiration of places and organizations: {", ".join(random.sample(NOUNS, 5))}!
        
        #TODO: the nouns might need some changing, e.g, prompt a model to also design a word for location or organization with those nouns.
        
        print(f"\n=== Sample {i} - {title} ===")

        completion = generate_completion(model, tokenizer, device, user_prompt, max_new_tokens=500, temperature=0.5)
        
        # If the model tries to echo your prompt, sometimes you can post-process 
        # to remove any repeated chunk from the beginning of the output.
        # For instance, if completion starts with the prompt, we can strip it out.
        # But first, let's see if it behaves.
        original_passage = sentences
        rewritten_passage = extract_rewritten_passages(completion)[0]

        
        # Create a data directory if it doesn't exist
        output_dir = os.path.join(os.path.dirname(__file__), '..', 'output')
        os.makedirs(output_dir, exist_ok=True)

        output_file = os.path.join(output_dir, 'contextual_articles.json')

        sample_result = {
            "title": title,
            "original": original_passage,
            "rewritten": rewritten_passage
        }
        # print("Entire prompt")
        # print(completion)
        print("Sample Result:")
        print(json.dumps(sample_result, indent=4))
        print("-" * 80)

        print(f"Saving results to: {output_file}")
        with open(output_file, 'w') as f:
            json.dump(sample_result, f, indent=4)

if __name__ == "__main__":
    sample_file = "MIND/auto-labeled/wiki/wiki_test.json"
    samples = load_samples(sample_file, 50)
    test_prompts(samples)
