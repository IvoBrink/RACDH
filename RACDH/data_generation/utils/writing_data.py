import json
import random
import os
from RACDH.config import params
import os
import json

def write_to_json(filename, data, ignore_dirs=False, overwrite=True):
    if ignore_dirs:
        output_path = params.output_path
    else:
        output_path = os.path.join(params.output_path, params.target_name, params.instruct_name)

    if not os.path.exists(output_path):
        os.makedirs(output_path)
        if params.debug:
            print(f"Created directory: {output_path}")

    full_path = os.path.join(output_path, filename)

    existing_data = []
    if os.path.exists(full_path) and not overwrite:
        with open(full_path, 'r') as file:
            try:
                existing_data = json.load(file)
                if params.debug:
                    print(f"Loaded existing data from {full_path}")
            except json.JSONDecodeError:
                existing_data = []
                if params.debug:
                    print(f"Error decoding JSON from {full_path}, starting from scratch.")

    # existing_titles = [d['title'] for d in existing_data]
    # new_data = [d for d in data if d['title'] not in existing_titles]
    existing_data.extend(data)

    if params.debug:
        print(f"Total data to be written: {len(existing_data)}")
        print(f"New data added: {len(data)}")

    with open(full_path, 'w') as file:
        json.dump(existing_data, file, indent=4)
        if params.debug:
            print(f"Data written to {full_path}")



def write_tensors(filename, data, ignore_dirs=False):
    import torch
    if ignore_dirs:
        output_path = params.output_path
    else:
        output_path = params.output_path + params.target_name + "/" + params.instruct_name + "/"
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    torch.save(data, output_path + filename)