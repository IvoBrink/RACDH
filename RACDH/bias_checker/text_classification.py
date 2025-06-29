import sys
import os
from tqdm import tqdm
sys.path.append(os.path.abspath("/home/ibrink/RACDH/RACDH/"))
import warnings

# Ignore all FutureWarnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from RACDH.data_generation.utils.reading_data import load_json
from RACDH.data_generation.utils.print import *
from RACDH.config import params
from RACDH.data_generation.utils.writing_data import write_to_json
from RACDH.bias_checker.logistic_regression import classify_texts


def get_last_sentence(text):
    """
    Extract the last sentence from a text string.
    
    Args:
        text (str): Input text
        
    Returns:
        str: Last sentence from the text
    """
    # Split on common sentence endings (., !, ?)
    sentences = text.replace('!', '.').replace('?', '.').split('.')
    
    # Remove empty strings and whitespace
    sentences = [s.strip() for s in sentences if s.strip()]
    
    # Return last sentence if exists, empty string if no sentences
    return sentences[-1] if sentences else ""

def remove_last_sentence(text):
    """
    Remove the last sentence from a text string.
    
    Args:
        text (str): Input text
        
    Returns:
        str: Text with last sentence removed
    """
    # Split on common sentence endings (., !, ?)
    sentences = text.replace('!', '.').replace('?', '.').split('.')
    
    # Remove empty strings and whitespace
    sentences = [s.strip() for s in sentences if s.strip()]
    
    # Return all but last sentence if multiple exist, empty string if 0-1 sentences
    return '. '.join(sentences[:-1]) + '.' if len(sentences) > 1 else ""


if __name__ == "__main__":
    samples = load_json(f"{params.target_name}/{params.instruct_name}/hiddens_metadata_all_2.json")

    n_synonym = len([x for x in samples if x["similar_entity"]])
    context = len([x for x in samples if x["similar_entity"] and x["label"] == "contextual"])
    parametric = len([x for x in samples if x["similar_entity"] and x["label"] == "parametric"])
    print(f"Number of similair entities in data {round((n_synonym/len(samples))*100,2)}%")
    print(f"Percentage that is contextual {round((context/n_synonym)*100,2)}%")
    print(f"Percentage that is parametric {round((parametric/n_synonym)*100,2)}%")

    if params.last_sentence:
        passages = [get_last_sentence(sample["passage"]) for sample in samples]
    else:
        passages = [sample["passage"] for sample in samples]

    labels = [sample["label"] for sample in samples]

    classify_texts(passages, labels, imbalance_mode="balanced", model_name="microsoft/deberta-v3-large")