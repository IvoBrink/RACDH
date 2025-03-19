import json
import random
import os
import torch
from RACDH.config import params


def write_to_json(filename, data, ignore_dirs=False):
    if ignore_dirs:
        output_path = params.output_path
    else:
        output_path = params.output_path + params.target_name + "/" + params.instruct_name + "/"

    if not os.path.exists(output_path):
        os.makedirs(output_path)
    with open(output_path + filename, "w") as file:
        json.dump(data, file, indent=4)


def write_tensors(filename, data, ignore_dirs=False):
    if ignore_dirs:
        output_path = params.output_path
    else:
        output_path = params.output_path + params.target_name + "/" + params.instruct_name + "/"
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    torch.save(data, output_path + filename)