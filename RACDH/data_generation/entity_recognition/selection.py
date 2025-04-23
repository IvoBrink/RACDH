import re
from collections import defaultdict
import numpy as np
from RACDH.data_generation.instruct_model import generate_completion_GPT
from RACDH.data_generation.cross_encoder import get_similarity_score


def is_purely_numeric(s):
    """Returns True if s (after stripping space/punctuation) is entirely digits."""
    s_stripped = re.sub(r"[^\w]", "", s)
    return s_stripped.isdigit()

def tokenize_and_normalize(s, min_token_length=3):
    """Split on whitespace after lowercasing and removing punctuation.
       Keep tokens of length >= min_token_length."""
    s = re.sub(r"[^\w\s]", " ", s.lower())
    tokens = [t for t in s.split() if len(t) >= min_token_length]
    return set(tokens)

def too_much_overlap_texts(text1, text2, sim_threshold=0.5):
    """
    Return True if text1 and text2 have enough overlap to warrant exclusion.
    1) If both texts are purely numeric, exclude only if they are identical.
    2) Check for subset relationships and the ratio of overlapping tokens.
    3) Utilize Cross Encoder embeddings to assess final similarity.
    """
    # 1) Numeric short-circuit
    if is_purely_numeric(text1) and is_purely_numeric(text2):
        return (text1 == text2)  # exclude only if exactly the same digits

    # 2) Token-based checks
    tokens_t = tokenize_and_normalize(text1)
    tokens_e = tokenize_and_normalize(text2)

    # (a) Subset check
    if tokens_e and tokens_t:  # both sets non-empty
        if tokens_e.issubset(tokens_t) or tokens_t.issubset(tokens_e):
            return True

    # DEPRECATED
    # # 3) Embedding similarity check
    # emb1 = embed_text(text1)
    # emb2 = embed_text(text2)
    # cos_sim = cosine_similarity(emb1, emb2)

    return get_similarity_score(text1, text2) >= sim_threshold

def entity_occurences(entities):
    """
    Returns a dictionary with the number of overlapping occurrences
    for each entity. If an entity has >1 overlap, we can consider it 'too frequent'.
    """
    occurrence_dict = {}

    for entity1, offset1 in entities:
        occurrence_dict[entity1] = 0
        for entity2, offset2 in entities:
            if offset1 == offset2:
                continue

            # You can tweak these thresholds as you like
            if too_much_overlap_texts(entity1, entity2, sim_threshold=0.5):
                occurrence_dict[entity1] += 1
    return occurrence_dict


def LLM_select(text, entities):
    prompt = f""" Given a Wikipedia passage and a list of mentioned entities, you are tasked to narrow down the list entities to a maximum of three using the following criteria.
1. Remove entities that are refer to a number, e.g., "first, "last", "twelve", "fourth".
2. Remove entities that refer to a very specific date, e.g., a range, months, days etc. Just a year is acceptable.
3. Remove entities that refer to any quantity.
4. The resulting entities should contain both well-known examples and lesser-known examples.


The string of the remaining entities must be exactly the same.

The passage:
{text}
The entities:
{[entity for entity, _, status in entities if status == "Correct"]}
Maximum of three selected entities:
"""
    str_entities = generate_completion_GPT(prompt, debug=False)
    # Convert the string representation of the list to an actual list
    selected_entities_list = str_entities.strip("[]").replace("'", "").split(", ")
    return selected_entities_list
    

    

def select_best_entities(text, entities, title):
    """
    Decide whether to redact an entity due to:
    1) Overlapping with the title
    2) Being too frequent (overlaps with multiple entities)
    3) Otherwise correct
    """
    selected_entities = []
    occurrences = entity_occurences(entities)

    for entity, offset in entities:
        # Check overlap with title
        if too_much_overlap_texts(title, entity):
            selected_entities.append((entity, offset, "Redacted due to title overlap"))

        # Check how often it overlaps with other entities
        elif occurrences[entity] > 1:
            selected_entities.append((entity, offset, "Redacted too frequent"))

        else:
            selected_entities.append((entity, offset, "Correct"))

    GPT_entities = LLM_select(text, selected_entities)

    for entity, offset, status in selected_entities:
        if status == "Correct":
            if entity in GPT_entities:
                selected_entities[selected_entities.index((entity, offset, status))] = (entity, offset, "Correct")
            else:
                selected_entities[selected_entities.index((entity, offset, status))] = (entity, offset, "GPT incorrect")

    return selected_entities
