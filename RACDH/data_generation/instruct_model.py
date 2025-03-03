from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList
import torch
from tqdm import tqdm
import re
from RACDH.config import params
from openai import OpenAI
client = OpenAI()

instruct_model_name_or_path = "meta-llama/Llama-3.1-8B-Instruct"
device = "cuda" if torch.cuda.is_available() else "cpu"

instruct_tokenizer = AutoTokenizer.from_pretrained(instruct_model_name_or_path)
instruct_model = AutoModelForCausalLM.from_pretrained(
    instruct_model_name_or_path,
    torch_dtype=torch.bfloat16
).to(device)


class StopOnMultipleStr(StoppingCriteria):
    def __init__(self, stop_strings, tokenizer):
        """
        stop_strings: list of strings that should trigger stopping
        tokenizer: a tokenizer to encode those strings into token-IDs
        """
        super().__init__()
        self.tokenizer = tokenizer

        # Pre-encode each variant of the stop string (no special tokens).
        self.stop_ids_list = []
        for s in stop_strings:
            ids = self.tokenizer.encode(s, add_special_tokens=False)
            self.stop_ids_list.append(ids)

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs):
        """
        Return True if the end of input_ids matches any of the pre-encoded
        stop_ids sequences.
        """
        sequence = input_ids[0].tolist()  # For a single-batch scenario

        for stop_ids in self.stop_ids_list:
            if len(sequence) >= len(stop_ids):
                # Compare the tail of input_ids to one of our stop sequences
                if sequence[-len(stop_ids):] == stop_ids:
                    print
                    return True
        return False
    

def extract_rewritten_passages(text, pattern):
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
    pattern = re.compile(pattern, flags=re.DOTALL
    )

    matches = re.findall(pattern, text)
    # Clean up extra whitespace:
    cleaned_passages = [m.strip() for m in matches]
    assert len(cleaned_passages) == 1, "More or less than one passage were extracted"
    return cleaned_passages[0]


def generate_completion_GPT(prompt, debug=False):
    model = "gpt-4o"
    completion = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a helpful data generator. You answer exactly as instructed using the information given."},
            {
                "role": "user",
                "content": prompt
            }
        ]
    )
    if debug:
        print("-" * 15 + f"{model}: Debugging output" + "-" * 15)
        print(completion.choices[0].message.content)
        print("-" * 30)
    return completion.choices[0].message.content

def generate_completion(prompt, pattern, max_new_tokens=256, temperature=0.5, debug=False):
    if params.OpenAI:
        return generate_completion_GPT(prompt, debug)
    else:
        return generate_completion_local(prompt, pattern, max_new_tokens, temperature, debug)


def generate_completion_local(prompt, pattern, max_new_tokens, temperature, debug):

    inputs = instruct_tokenizer(prompt, return_tensors="pt").to(device)

    possible_variants = [
    ">> ",
    " >>"
    ">>>",        
    " >>>",       
    ">>> ",       
    "\n>>>",      
    ]

    stop_criteria = StoppingCriteriaList([StopOnMultipleStr(possible_variants, instruct_tokenizer)])

    output_ids = instruct_model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        do_sample=True,
        top_p=0.9,
        pad_token_id=instruct_tokenizer.eos_token_id,
        eos_token_id=instruct_tokenizer.eos_token_id,
        stopping_criteria=stop_criteria
    )
    text = instruct_tokenizer.decode(output_ids[0], skip_special_tokens=True)
    if debug:
        print("-" * 15 + f"{instruct_model_name_or_path}: Debugging entire prompt + output" + "-" * 15)
        print(text)
        print("-" * 30)
    
    return extract_rewritten_passages(text, pattern)