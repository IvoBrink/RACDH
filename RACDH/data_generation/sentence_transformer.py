from sentence_transformers import SentenceTransformer
from RACDH.config import params
import numpy as np

############################
# Setup a Sentence Transformer model
############################
sentence_model = SentenceTransformer('all-MiniLM-L6-v2')


def embed_text(text: str):
    """Compute the sentence embedding for a string."""
    # We use `convert_to_numpy=True` for easy numeric handling:
    return sentence_model.encode([text], convert_to_numpy=True)[0]


def cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """Compute cosine similarity between two 1-D numpy arrays."""
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)
    if norm_a == 0.0 or norm_b == 0.0:
        # Edge case: zero vector
        return 0.0
    return np.dot(vec_a, vec_b) / (norm_a * norm_b)


def similar_text(text1, text2):
    embed1, embed2 = embed_text(text1), embed_text(text2)
    similarity_score = cosine_similarity(embed1, embed2)
    print(similarity_score)
    return similarity_score >= params.similarity_threshold_entity