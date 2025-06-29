from RACDH.data_generation.inference.find_similar_entities import find_similar_entities
from RACDH.config import params

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



# ---------- helpers -------------------------------------------------- #
def _sort_tokens(token_hiddens):
    """Return [(step, tok_str, hidden), …] sorted by generation step."""
    return sorted(
        [(step, t_str, hid) for (step, t_str), hid in token_hiddens.items()],
        key=lambda x: x[0]
    )


def _reconstruct_text(tokens):
    """
    From tokens → (full_text, char_offsets)
    where char_offsets[i] == (start_char, end_char) of tokens[i].
    """
    full_text, offsets, pos = "", [], 0
    for _, t_str, _ in tokens:
        start, end = pos, pos + len(t_str)
        full_text += t_str
        offsets.append((start, end))
        pos = end
    return full_text, offsets


def _locate_entity(full_text, entity, allow_fuzzy):
    """
    Return (idx_start, idx_end, resolved_entity, used_fuzzy).
    If not found (even with fuzzy) → (None, None, entity, False).
    """
    full_lower, ent_lower = full_text.lower(), entity.lower()
    idx = full_lower.find(ent_lower)
    if idx != -1:
        return idx, idx + len(entity), entity, False

    if not allow_fuzzy:
        return None, None, entity, False

    similar = find_similar_entities(full_text, entity)
    if similar is None:
        return None, None, entity, False

    ent_lower = similar.lower()
    idx = full_lower.find(ent_lower)
    if idx == -1:          # extremely unlikely guard
        return None, None, entity, False

    return idx, idx + len(similar), similar, True


def _overlapping_token_indices(offsets, span):
    """Indices of tokens whose char-range overlaps [span_start, span_end)."""
    span_start, span_end = span
    return [
        i for i, (s, e) in enumerate(offsets)
        if not (e <= span_start or s >= span_end)
    ]


def _build_result(tokens, idxs, first_token_gen, before_ent_hidden, similar_flag):
    """Assemble final dictionary."""
    first_idx, last_idx = idxs[0], idxs[-1]
    matched = tokens[first_idx:last_idx + 1]

    return {
        "tokens": [
            {"step": step, "token_str": t_str, "hidden": hid}
            for (step, t_str, hid) in matched
        ],
        "first_token_entity": matched[0][2],
        "last_token_entity":  matched[-1][2],
        "first_token_generation": first_token_gen,
        "last_token_before_entity": before_ent_hidden if not None else matched[0][2],
        "similar_entity": similar_flag,
    }


# ---------- public API ----------------------------------------------- #
def get_entity_span_text_align(token_hiddens,
                               entity,
                               similarity_check_inference=params.similarity_check_inference):
    """
    Locate `entity` in generated text and return hidden-state stacks
    for relevant tokens.

    See `_build_result` for returned structure.
    """
    # 1. tokens sorted + first token of entire generation
    tokens = _sort_tokens(token_hiddens)
    if not tokens:
        return {}

    first_token_gen_hidden = tokens[0][2]

    # 2. full string and char offsets
    full_text, offsets = _reconstruct_text(tokens)

    # 3. locate entity (with optional fuzzy fallback)
    idx, end_idx, resolved_entity, used_fuzzy = _locate_entity(
        full_text, entity, similarity_check_inference)
    if idx is None:
        return {}

    # 4. which token indices overlap that span?
    matched_indices = _overlapping_token_indices(offsets, (idx, end_idx))
    if not matched_indices:
        return {}

    # 5. hidden state just before the entity (may be None)
    before_ent_hidden = (
        tokens[matched_indices[0] - 1][2] if matched_indices[0] > 0 else None
    )

    # 6. build and return
    return _build_result(
        tokens, matched_indices,
        first_token_gen_hidden,
        before_ent_hidden,
        used_fuzzy
    )

