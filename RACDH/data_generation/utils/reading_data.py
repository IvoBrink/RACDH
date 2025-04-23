import json
import random
import os
from RACDH.config import params

def load_samples_wiki(filename="wiki_train.json", n_samples=None, check_earlier_created_data = False):
    path = params.wiki_path + filename
    with open(path, 'r') as f:
        data = json.load(f)
    if params.debug:print(f"Data length of {filename}: {len(data)}")
    
    # Filter out passages with low character count
    filtered_data = [sample for sample in data if len(" ".join(sample['sentences'])) > 200]

    if check_earlier_created_data:
        existing_data = load_json(f"extracted_entities.json")
        existing_titles = [data['title'] for data in existing_data]
        filtered_data_existing = [sample for sample in filtered_data if sample['title'] not in existing_titles]
        if params.debug:
            excluded_data_count = len(filtered_data) - len(filtered_data_existing)
            print(f"Excluded data count because of existing data: {excluded_data_count}")
        filtered_data = filtered_data_existing
        
    if params.debug:print(f"After all filtering data length of {filename}: {len(filtered_data)}")
    if n_samples is not None:
        return random.sample(filtered_data, n_samples)
    else:
        return filtered_data


def load_samples_new_wiki(filename="random_wiki_articles.ndjson", n_samples=None):
    path = os.path.join(params.output_path, filename)

    with open(path, "r", encoding="utf-8") as fh:
        data = [json.loads(line) for line in fh if line.strip()]

    if n_samples is not None:
        n_samples = min(n_samples, len(data))
        return random.sample(data, n_samples)

    return data
    

def load_json(filename, n_samples=None, existing_data_file = None):
    path = params.output_path + filename
    with open(path, 'r') as f:
        data = json.load(f)
    if params.debug:print(f"Data length of {filename}: {len(data)}")

    if existing_data_file is not None:
        existing_data = load_json(existing_data_file)
        existing_titles = [d['title'] for d in existing_data]
        filtered_data = [d for d in data if d['title'] not in existing_titles]
        if params.debug:
            excluded_data_count = len(data) - len(filtered_data)
            print(f"Excluded data count because of existing data: {excluded_data_count}")
        data = filtered_data

    if n_samples is not None:
        return random.sample(data, n_samples)
    else:
        return data
