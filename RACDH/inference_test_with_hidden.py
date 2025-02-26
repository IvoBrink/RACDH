from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Cashed models: meta-llama/Llama-3.1-8B-Instruct, deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B 

# model_name_or_path = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"  # example name
model_name_or_path = "meta-llama/Llama-3.1-8B"  # example name
# model_name_or_path = "mistralai/Mistral-7B-v0.3"
# model_name_or_path = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"

device = "cuda" if torch.cuda.is_available() else "cpu"

# Download/check the model & tokenizer from the Hugging Face hub into your local cache
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
model = AutoModelForCausalLM.from_pretrained(
    model_name_or_path,
    torch_dtype=torch.bfloat16,
    output_attentions=True,
    output_hidden_states=True
).to(device)

# prompt_text = "The Battle of Evesham ( 4 August 1265 ) was one of the two main battles of 13th century England 's Second Barons ' War . It marked the defeat of Simon de Montfort , Earl of Leicester , and the rebellious barons by Prince Edward – later King Edward I – who led the forces of his father , King Henry III . It took place on 4 August 1265 , near the town of Evesham , Worcestershire . "
prompt_text = """Alice: I can’t remember exactly who was the king of England in 1265 during the Battle of Evesham . I can't remember.
Bob: Actually, I know. It was """
inputs = tokenizer(prompt_text, return_tensors="pt").to(device)

generated_ids = inputs["input_ids"]
attention_mask = inputs["attention_mask"]

max_new_tokens = 15  # just a small number of tokens for demonstration
for step in range(max_new_tokens):
    # 1) Forward pass
    outputs = model(
        input_ids=generated_ids,
        attention_mask=attention_mask,
        output_attentions=True,
        output_hidden_states=True
    )
    
    # 2) Get the logits for the last position
    next_token_logits = outputs.logits[:, -1, :]  # shape: [batch_size, vocab_size]
    
    # 3) Convert logits to probabilities for the top-k
    #    (if you're just picking by argmax, you don't need the softmax, but let's do it for top-k analysis)
    probabilities = torch.softmax(next_token_logits, dim=-1)
    
    topk_probs, topk_indices = torch.topk(probabilities, k=3, dim=-1)
    topk_probs = topk_probs[0]    # shape [3]
    topk_indices = topk_indices[0]  # shape [3]
    
    print(f"\n=== Decoding step {step+1} ===")
    print("Top 3 most likely tokens:")
    for rank in range(3):
        token_id = topk_indices[rank].item()
        token_prob = topk_probs[rank].item()
        token_str = tokenizer.decode([token_id])
        print(f"  {rank+1}) '{token_str}' (id={token_id}, prob={token_prob:.4f})")
    
    # 4) Pick the actual next token via greedy argmax (you could also do sampling)
    next_token_id = torch.argmax(next_token_logits, dim=-1, keepdim=True)
    chosen_str = tokenizer.decode(next_token_id[0])
    print(f"Chosen token: '{chosen_str}' (token_id={next_token_id.item()})")
    
    # 5) Get final-layer attention & hidden states
    final_layer_attention = outputs.attentions[-1]        # [batch_size, num_heads, seq_len, seq_len]
    final_layer_hidden = outputs.hidden_states[-1]        # [batch_size, seq_len, hidden_dim]
    
    # Print summary statistics for attention & hidden states
    print(f"Final-layer attention shape: {final_layer_attention.shape}")
    print(f"  Mean attention: {final_layer_attention.mean().item():.4f}")
    print(f"  Min attention:  {final_layer_attention.min().item():.4f}")
    print(f"  Max attention:  {final_layer_attention.max().item():.4f}")
    print(f"Final-layer hidden shape: {final_layer_hidden.shape}")
    print(f"  Mean hidden: {final_layer_hidden.mean().item():.4f}")
    print(f"  Min hidden:  {final_layer_hidden.min().item():.4f}")
    print(f"  Max hidden:  {final_layer_hidden.max().item():.4f}")
    
    # 6) Append the chosen token to the sequence
    generated_ids = torch.cat([generated_ids, next_token_id], dim=-1)
    
    # 7) Update the attention mask
    new_mask = torch.ones(
        (attention_mask.size(0), 1),
        dtype=attention_mask.dtype,
        device=attention_mask.device
    )
    attention_mask = torch.cat([attention_mask, new_mask], dim=1)

# Finally, decode the complete sequence
final_output = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
print("\n=== Final Decoded Text ===")
print(final_output)
