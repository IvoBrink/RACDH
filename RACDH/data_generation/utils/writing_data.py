import json
import random
import os
from RACDH.config import params


def write_to_json(filename, data):
    if not os.path.exists(params.output_path):
        os.makedirs(params.output_path)
    with open(params.output_path + filename, "w") as file:
        json.dump(data, file, indent=4)