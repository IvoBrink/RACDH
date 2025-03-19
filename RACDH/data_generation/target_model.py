import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, StoppingCriteria, StoppingCriteriaList
from RACDH.data_generation.utils.print import *

taget_model_name_or_path = params.taget_model_name_or_path# example name

torch.cuda.empty_cache()
device = "cuda" if torch.cuda.is_available() else "cpu"

# Download/check the model & tokenizer from the Hugging Face hub into your local cache
if 'taget_model' not in locals():
    target_tokenizer = AutoTokenizer.from_pretrained(taget_model_name_or_path)

# Load model only if not already loaded
if 'taget_model' not in locals():
    taget_model = AutoModelForCausalLM.from_pretrained(
        taget_model_name_or_path,
        torch_dtype=torch.bfloat16
    ).to(device)



class TokenMatchStop(StoppingCriteria):
    """
    Stops generation if the *last generated token* is exactly in `stop_tokens`.
    Note this compares *token strings* as returned by `tokenizer.convert_ids_to_tokens()`,
    not decoded text fragments.
    """
    def __init__(self, tokenizer, stop_tokens):
        super().__init__()
        self.tokenizer = tokenizer
        # Put your stop token strings in a set for quick membership checks
        self.stop_tokens = set(stop_tokens)

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        # The model generates one token at a time, so check the *newly generated token* (the last one)
        last_token_id = input_ids[0, -1].item()
        # Convert just that single ID back to its token string
        last_token_str = self.tokenizer.convert_ids_to_tokens([last_token_id])[0]
        
        # Return True if it matches one of our designated stop tokens
        return last_token_str in self.stop_tokens



def generate_completion(prompt, entity, max_new_tokens=25, temperature=0.5, debug=False):
    inputs = target_tokenizer(prompt, return_tensors="pt").to(device)


    stop_tokens_without_entity_consideration = ['.', '!', '?', '.Ċ', 'Ċ', ":", "Q:", "ĊQ", "Alice", "Alice:", "Alice: ", ":"]
    stop_tokens = [x for x in stop_tokens_without_entity_consideration if x not in entity]

    custom_stop = TokenMatchStop(target_tokenizer, stop_tokens)
    stopping_criteria = StoppingCriteriaList([custom_stop])

    output_ids = taget_model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        do_sample=True,
        top_p=0.9,
        pad_token_id=target_tokenizer.eos_token_id,
        eos_token_id=target_tokenizer.eos_token_id,
        stopping_criteria=stopping_criteria 
    )

     # 1) Get the prompt length in tokens
    prompt_length = inputs["input_ids"].shape[1]
    # 2) Slice out only the newly generated tokens
    generated_ids = output_ids[0][prompt_length:]
    # 3) Decode just that portion
    generated_text = target_tokenizer.decode(generated_ids, skip_special_tokens=True)
    # generated_tokens = target_tokenizer.convert_ids_to_tokens(generated_ids)

    output = clean_output(generated_text)

    if debug:

        print_h3(f"{taget_model_name_or_path} output (cleaned!)")
        print_generated_completion(prompt," " + output) #Note: here we use the cleaned output, so we cannot check if stopping crit. works
        # print_h3("Tokens:")
        # print(generated_tokens)
        # print_h3("Cleaned output")
        # print(output)
    
    return prompt, output



def clean_output(output):
    # Remove everything after a newline and strip whitespace
    split = output.split('\n')
    output = split[0].strip()
    if len(split) > 1:
        print_warning(f"Number of newline splits is more than one: {split}")
    # Remove punctuation at the end of the sentence
    return output.rstrip('.,!?;:')  # Add any other punctuation you want to remove


#### Code used for getting topk stuff ####
# probabilities = torch.softmax(next_token_logits, dim=-1)
# topk_probs, topk_indices = torch.topk(probabilities, k=3, dim=-1)
# topk_probs = topk_probs[0]    # shape [3]
# topk_indices = topk_indices[0]  # shape [3]
# 
# print(f"\n=== Decoding step {step+1} ===")
# print("Top 3 most likely tokens:")
# for rank in range(3):
#     token_id = topk_indices[rank].item()
#     token_prob = topk_probs[rank].item()
#     token_str = target_tokenizer.decode([token_id])
#     print(f"  {rank+1}) '{token_str}' (id={token_id}, prob={token_prob:.4f})")

def generate_completion_extract_hiddens(prompt, max_new_tokens=25, temperature=0.5, debug=False):
    inputs = target_tokenizer(prompt, return_tensors="pt").to(device)
    generated_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    
    # Dictionary to store (step, token) -> (attention_row, hidden_vec)
    token_info = {}

    for step in range(max_new_tokens):
        outputs = taget_model(
            input_ids=generated_ids,
            attention_mask=attention_mask,
            output_attentions=True,
            output_hidden_states=True
        )
        
        next_token_logits = outputs.logits[:, -1, :]
        
        # Greedy argmax for simplicity (you could sample for temperature)
        next_token_id = torch.argmax(next_token_logits, dim=-1, keepdim=True)
        chosen_str = target_tokenizer.decode(next_token_id[0])
        
        # Get final-layer attention and hidden states
        final_layer_attention = outputs.attentions[-1]   # [batch_size, n_heads, seq_len, seq_len]
        final_layer_hidden = outputs.hidden_states[-1]   # [batch_size, seq_len, hidden_dim]
        
        # Slice out just the row and hidden vector for the newly generated token
        # final_layer_attention[:, :, -1, :] => shape: [batch_size, n_heads, 1, seq_len]
        # final_layer_hidden[:, -1, :]       => shape: [batch_size, hidden_dim]
        step_attention = final_layer_attention[:, :, -1, :].detach().cpu()
        step_hidden = final_layer_hidden[:, -1, :].detach().cpu()
        
        token_info[(step, chosen_str)] = (step_attention, step_hidden)
        
        # Append the chosen token to the sequence
        generated_ids = torch.cat([generated_ids, next_token_id], dim=-1)
        
        # Update the attention mask
        new_mask = torch.ones(
            (attention_mask.size(0), 1),
            dtype=attention_mask.dtype,
            device=attention_mask.device
        )
        attention_mask = torch.cat([attention_mask, new_mask], dim=1)

    return token_info

