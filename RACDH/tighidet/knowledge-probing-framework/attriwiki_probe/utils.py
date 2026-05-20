from collections import defaultdict
from sklearn.model_selection import train_test_split
import torch
import numpy as np
import random, numpy as np, torch
from typing import Any, Dict, List, Tuple
from sklearn.model_selection import GroupShuffleSplit

def seed_everything(seed: int = 42) -> None:
    """
    Make all common Python, NumPy and PyTorch sources of randomness reproducible.
    """
    random.seed(seed)                      
    np.random.seed(seed)                    
    torch.manual_seed(seed)                
    torch.cuda.manual_seed_all(seed)        

    # extra safety for deterministic kernels
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def make_title_aware_val_split(
    train_idx: np.ndarray,
    meta: List[Dict[str, Any]],
    *,
    val_frac: float = 0.15,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    titles = np.array([meta[i]["title"] for i in train_idx])
    splitter = GroupShuffleSplit(n_splits=1, test_size=val_frac, random_state=seed)
    inner_mask, val_mask = next(splitter.split(train_idx, groups=titles))
    return train_idx[inner_mask], train_idx[val_mask]

def groupwise_train_test_split(meta_hidden_states,
                               test_size: float = 0.2,
                               random_state: int = 42):
    """Split *titles* into train/test then map back to sample indices."""
    title_to_indices: dict[str, list[int]] = defaultdict(list)
    for idx, sample in enumerate(meta_hidden_states):
        title_to_indices[sample.get("title", "")].append(idx)

    unique_titles = list(title_to_indices)
    train_titles, test_titles = train_test_split(unique_titles,
                                                test_size=test_size,
                                                random_state=random_state)

    train_idx = [i for t in train_titles for i in title_to_indices[t]]
    test_idx = [i for t in test_titles for i in title_to_indices[t]]
    return train_idx, test_idx



def load_vectors_and_labels(hidden_states,
                            meta_hidden_states,
                            token_key: str = "first_token_entity",
                            reduce: str = "stack"):
    # … same docstring / checks …

    vecs, labels, flags, titles = [], [], [], []

    for sample in meta_hidden_states:
        idx   = sample["hidden_states_index"]
        stack = hidden_states[idx][token_key].detach().cpu().to(torch.float32)
        # --------------------------------------------------- #
        #  CHANGED: keep the tensor for "stack", convert only
        #           for "last"
        # --------------------------------------------------- #
        if reduce == "stack":
            vec = stack                                    # Tensor (L, H)
        else:  # "last"
            vec = stack[-1].numpy().astype("float32")      # np.ndarray (H,)

        vecs.append(vec)
        labels.append(1 if sample["label"] == "contextual" else 0)
        flags .append(sample.get("similar_entity", False))
        titles.append(sample.get("title", ""))

    # ---------------- final assemble ------------------------
    if reduce == "stack":
        X = torch.stack(vecs)                              # Tensor (N, L, H)
        y = torch.tensor(labels, dtype=torch.long)
    else:  # "last"
        X = np.asarray(vecs, dtype=np.float32)             # (N, H)
        y = np.asarray(labels, dtype=int)

    flags  = np.asarray(flags, dtype=bool)
    titles = np.asarray(titles)

    return X, y, flags, titles