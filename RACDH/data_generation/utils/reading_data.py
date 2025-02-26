import json
import random

def load_samples(filepath="wiki_train.json", n_samples=None):
    path = "/home/ibrink/RACDH/RACDH/MIND/auto-labeled/wiki/" + filepath
    with open(path, 'r') as f:
        data = json.load(f)
    
    if n_samples is not None:
        return random.sample(data, n_samples)
    else:
        return data