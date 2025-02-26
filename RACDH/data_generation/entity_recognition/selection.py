import re
from collections import defaultdict

import numpy as np
from sentence_transformers import SentenceTransformer

############################
# Setup a Sentence Transformer model
############################
# You can choose another model if you want
model = SentenceTransformer('all-MiniLM-L6-v2')

############################
# Embedding + Cosine Similarity
############################
def embed_text(text: str):
    """Compute the sentence embedding for a string."""
    # We use `convert_to_numpy=True` for easy numeric handling:
    return model.encode([text], convert_to_numpy=True)[0]

def cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """Compute cosine similarity between two 1-D numpy arrays."""
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)
    if norm_a == 0.0 or norm_b == 0.0:
        # Edge case: zero vector
        return 0.0
    return np.dot(vec_a, vec_b) / (norm_a * norm_b)


############################
# Utilities
############################
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

############################
# Overlap Check with SentenceTransformer for final step
############################
def too_much_overlap_texts(text1, text2, 
                           token_threshold=0.9, 
                           sim_threshold=0.7):
    """
    Return True if text1 & text2 'overlap enough' to exclude.
    1) If both purely numeric, only exclude if identical.
    2) Check subset & overlap ratio of tokens.
    3) Use SentenceTransformer embeddings for final similarity.
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

    # # (b) Overlap ratio
    # overlap = tokens_e.intersection(tokens_t)
    # smaller_set_size = min(len(tokens_e), len(tokens_t)) or 1
    # overlap_ratio = len(overlap) / smaller_set_size
    # if overlap_ratio >= token_threshold:
    #     return True

    # 3) Embedding similarity check
    emb1 = embed_text(text1)
    emb2 = embed_text(text2)
    cos_sim = cosine_similarity(emb1, emb2)

    return cos_sim >= sim_threshold


############################
# Entity Occurrences + Filtering
############################
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
            if too_much_overlap_texts(entity1, entity2, 
                                      token_threshold=0.5, 
                                      sim_threshold=0.5):
                occurrence_dict[entity1] += 1
    return occurrence_dict

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

    return selected_entities
