import random
import json
from transformers import StoppingCriteria, StoppingCriteriaList
import torch

def load_samples(filepath="wiki_train.json", n_samples=10):
    with open(filepath, 'r') as f:
        data = json.load(f)
    return random.sample(data, n_samples)



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


def generate_completion(model, tokenizer, device, prompt, max_new_tokens=256, temperature=0.5):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    possible_variants = [
    ">> ",
    " >>"
    ">>>",        
    " >>>",       
    ">>> ",       
    "\n>>>",      
    ]

    stop_criteria = StoppingCriteriaList([StopOnMultipleStr(possible_variants, tokenizer)])

    output_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        do_sample=True,
        top_p=0.9,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        stopping_criteria=stop_criteria
    )
    text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    # Cut off at ###
    # text = text.split("###")[-1]
    return text
