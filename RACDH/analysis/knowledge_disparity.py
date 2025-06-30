import sys
import os
import numpy as np
import pandas as pd
from scipy.stats import fisher_exact, chi2_contingency, norm
from tqdm import tqdm
sys.path.append(os.path.abspath("/home/ibrink/RACDH/RACDH/"))
import warnings
from collections import defaultdict
import json
import matplotlib.pyplot as plt
from matplotlib_venn import venn2


# Ignore all FutureWarnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from RACDH.data_generation.utils.reading_data import load_json
from RACDH.data_generation.utils.print import *
from RACDH.config import params


if __name__ == "__main__":
    llama_path  = "RACDH/data/Mistral-7B-v0.1/gpt-4o-mini/knowledge_2.json"
    mistral_path = "RACDH/data/Llama-3.1-8B/gpt-4o-mini/knowledge_2.json"
    with open(llama_path, "r") as f:
        llama_data = json.load(f)
    with open(mistral_path, "r") as f:
        mistral_data = json.load(f)

    # Build sets of all entity judgments
    entity_judgments = defaultdict(lambda: {"llama": 0, "mistral": 0})

    # Helper to fill judgments
    def collect_judgments(data, model_name):
        for item in data:
            for ent, val in item["known_entites"].items():
                if val == 1:
                    entity_judgments[ent][model_name] = 1
            for ent in item.get("ignored_entities", []):
                if ent not in entity_judgments:
                    entity_judgments[ent][model_name] = 0  # explicit 0

    collect_judgments(llama_data, "llama")
    collect_judgments(mistral_data, "mistral")

    # Compute 2x2 table
    A = B = C = D = 0
    for ent, status in entity_judgments.items():
        l = status["llama"]
        m = status["mistral"]
        if l == 1 and m == 1:
            A += 1  # Both know
        elif l == 1 and m == 0:
            B += 1  # Only LLaMA knows
        elif l == 0 and m == 1:
            C += 1  # Only Mistral knows
        else:
            D += 1  # Neither knows

    # Calculate total known and not known entities for each model
    total_llama_known = sum(1 for status in entity_judgments.values() if status["llama"] == 1)
    total_llama_not_known = sum(1 for status in entity_judgments.values() if status["llama"] == 0)
    total_mistral_known = sum(1 for status in entity_judgments.values() if status["mistral"] == 1)
    total_mistral_not_known = sum(1 for status in entity_judgments.values() if status["mistral"] == 0)

    # Print table
    print("\n2x2 Knowledge Overlap Table (Entities)")
    print("--------------------------------------")
    print(f"{'':>20} | {'Mistral: Known':>15} | {'Mistral: Not Known':>20}")
    print(f"{'LLaMA: Known':>20} | {A:>15} | {B:>20}")
    print(f"{'LLaMA: Not Known':>20} | {C:>15} | {D:>20}")
    print("\nTotal entities known and not known:")
    print(f"{'':>20} | {'Known':>10} | {'Not Known':>12}")
    print(f"{'LLaMA':>20} | {total_llama_known:>10} | {total_llama_not_known:>12}")
    print(f"{'Mistral':>20} | {total_mistral_known:>10} | {total_mistral_not_known:>12}")



         # Venn diagram with custom colors and full model names
    plt.figure(figsize=(6, 6))
    venn2(
        subsets=(B, C, A),  # (only_llama, only_mistral, both)
        set_labels=("LLaMA-3.1-8B Known", "Mistral-7B-v0.1 Known"),
        set_colors=('#ff339a', '#1c47e3'),  # Custom purples/blues
        alpha=1
    )
    plt.title("Overlap of Known Entities Between Models")
    plt.tight_layout()
    plt.savefig("RACDH/data/plots/knowledge_disparity.png")

