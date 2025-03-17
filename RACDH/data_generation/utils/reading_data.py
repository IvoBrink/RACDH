import json
import random
from RACDH.config import params

def load_samples_wiki(filename="wiki_train.json", n_samples=None):
    path = params.wiki_path + filename
    with open(path, 'r') as f:
        data = json.load(f)
    
    # Filter out passages with low character count
    filtered_data = [sample for sample in data if len(" ".join(sample['sentences'])) > 200]
    
    if n_samples is not None:
        return random.sample(filtered_data, n_samples)
    else:
        return filtered_data
    

def load_json(filename, n_samples=None):
    path = params.output_path + filename
    with open(path, 'r') as f:
        data = json.load(f)
    
    if n_samples is not None:
        return random.sample(data, n_samples)
    else:
        return data