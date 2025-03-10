import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from RACDH.data_generation.utils.print import *

taget_model_name_or_path = params.taget_model_name_or_path# example name


device = "cuda" if torch.cuda.is_available() else "cpu"

# Download/check the model & tokenizer from the Hugging Face hub into your local cache
target_tokenizer = AutoTokenizer.from_pretrained(taget_model_name_or_path)
taget_model = AutoModelForCausalLM.from_pretrained(
    taget_model_name_or_path,
    torch_dtype=torch.bfloat16
).to(device)


def generate_completion(prompt, max_new_tokens=25, temperature=0.5, debug=False):
    inputs = target_tokenizer(prompt, return_tensors="pt").to(device)

    output_ids = taget_model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        do_sample=True,
        top_p=0.9,
        pad_token_id=target_tokenizer.eos_token_id,
        eos_token_id=target_tokenizer.eos_token_id
    )

     # 1) Get the prompt length in tokens
    prompt_length = inputs["input_ids"].shape[1]
    # 2) Slice out only the newly generated tokens
    generated_ids = output_ids[0][prompt_length:]
    # 3) Decode just that portion
    generated_text = target_tokenizer.decode(generated_ids, skip_special_tokens=True)


    text = target_tokenizer.decode(output_ids[0], skip_special_tokens=True)
    if debug:
        print_h3(f"{taget_model_name_or_path} output")
        print_generated_completion(prompt, generated_text)
    
    return prompt, generated_text

