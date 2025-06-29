import sys
import os
import random
sys.path.append(os.path.abspath("/home/ibrink/RACDH/RACDH/"))
from RACDH.data_generation.utils.reading_data import load_json
from RACDH.data_generation.utils.writing_data import write_to_json
from RACDH.config import params

# def make_hashable(obj):
#     if isinstance(obj, (tuple, list)):
#         return tuple(make_hashable(item) for item in obj)
#     elif isinstance(obj, dict):
#         return tuple(sorted((k, make_hashable(v)) for k, v in obj.items()))
#     return obj

if __name__ == "__main__":
    root = f"{params.target_name}/{params.instruct_name}/"
    file = root + "completions_2.json"
    print("Loading samples...")
    samples = load_json(file)
    print(f"Loaded {len(samples)} samples")
    
    print("Removing duplicates...")
    unique_samples = []
    unique_ids = []
    parametric = 0
    contextual = 0
    for sample in samples:
        if sample["label"] == "parametric":
            parametric += 1
        else:
            contextual += 1
        if (sample["title"], sample["entity"]) in unique_ids:
            continue
        unique_samples.append(sample)
        unique_ids.append((sample["title"], sample["entity"]))
    
    print(f"Reduced to {len(unique_samples)} unique samples")

    print(f"Parametric: {parametric} out of {len(unique_samples)} ({parametric/len(unique_samples)*100:.2f}%)")
    print(f"Contextual: {contextual} out of {len(unique_samples)} ({contextual/len(unique_samples)*100:.2f}%)")

    # write_to_json(file, unique_samples, ignore_dirs=True, overwrite=True)
    print("Done!")
    


