
def reconstruct_generated_text(token_info):
    """
    Reconstructs the generated text exactly as the model produced it, 
    by sorting tokens by their generation step and concatenating 
    the token strings.
    """
    # Sort token_info keys by step
    sorted_keys = sorted(token_info.keys(), key=lambda x: x[0])  # (step, token_str)

    # Concatenate the token strings in generation order
    text_parts = [key[1] for key in sorted_keys]
    full_text = "".join(text_parts)

    return full_text



def get_entity_span_text_align(token_info, entity):
    """
    1) Reconstruct the final text from the generated tokens (in order).
    2) Find the FIRST substring match of `entity` in that text (case-insensitive).
    3) Identify which tokens overlap with that matched range.
    4) Return a dict with keys:
       - "tokens": list of (step, token_str, attention, hidden) for the matched span
       - "first_token_attention", "first_token_hidden"
       - "last_token_attention",  "last_token_hidden"

    If no match, return {} (an empty dict).
    """

    #TODO: do entity recognition and find entity, if present, that is similar.

    # 1) Sort by step, build a list of (step, token_str, attn, hid)
    sorted_items = sorted(token_info.items(), key=lambda x: x[0][0])
    tokens = [(step, t_str, attn, hid) for (step, t_str), (attn, hid) in sorted_items]

    # Reconstruct exact text & keep track of token char offsets
    full_text = ""
    token_char_offsets = []
    current_pos = 0

    for (step, t_str, attn, hid) in tokens:
        start_char = current_pos
        full_text += t_str
        end_char = start_char + len(t_str)
        token_char_offsets.append((start_char, end_char))
        current_pos = end_char

    # 2) Find the FIRST match of `entity` (ignore case)
    entity_lower = entity.lower()
    full_lower = full_text.lower()
    idx = full_lower.find(entity_lower)  # returns -1 if not found
    if idx == -1:
        # No match found
        return {}

    end_idx = idx + len(entity)

    # 3) Identify which tokens overlap with that matched range
    matched_indices = []
    for i, (start_char, end_char) in enumerate(token_char_offsets):
        # Overlap if not completely to the left or right
        if not (end_char <= idx or start_char >= end_idx):
            matched_indices.append(i)

    if not matched_indices:
        # No tokens actually overlapped (shouldn't happen if text truly matches)
        return {}

    first_token_idx = matched_indices[0]
    last_token_idx  = matched_indices[-1]

    # 4) Extract the token info from those indices
    matched_tokens = tokens[first_token_idx : last_token_idx + 1]

    # The very first token in the span
    _, _, first_token_attn, first_token_hidden = tokens[first_token_idx]
    # The very last token in the span
    _, _, last_token_attn, last_token_hidden   = tokens[last_token_idx]

    # Build a friendly return structure
    result = {
        "tokens": [
            {
                "step": step,
                "token_str": t_str,
                "attention": attn,
                "hidden": hid
            }
            for (step, t_str, attn, hid) in matched_tokens
        ],
        "first_token_attention": first_token_attn,
        "first_token_hidden": first_token_hidden,
        "last_token_attention": last_token_attn,
        "last_token_hidden": last_token_hidden,
    }

    return result
