import torch
import matplotlib.pyplot as plt
import numpy as np
import os

file_path = "RACDH/data/Llama-3.1-8B/gpt-4o-mini/hiddens_all_2.pt"
output_dir = "RACDH/data/plots/activations"
os.makedirs(output_dir, exist_ok=True)

token_keys = [
    "first_token_entity",
    "last_token_entity",
    "first_token_generation"
]

# Initialize sums and counts
layer_sums = {k: None for k in token_keys}
layer_counts = {k: 0 for k in token_keys}

# Streaming computation of layer norm means
for entry in torch.load(file_path, map_location="cpu"):
    for key in token_keys:
        data = entry[key]  # shape (L, D)
        norms = data.norm(dim=-1).to(torch.float32)
        if layer_sums[key] is None:
            layer_sums[key] = norms
        else:
            layer_sums[key] += norms
        layer_counts[key] += 1

# Normalize by count
layer_means = {
    k: (layer_sums[k] / layer_counts[k]).numpy()
    for k in token_keys
}

# Plot
for key, values in layer_means.items():
    plt.figure(figsize=(10, 4))
    plt.plot(values, marker="o")
    plt.title(f"Average Activation Norm per Layer – {key}")
    plt.xlabel("Layer Index")
    plt.ylabel("Mean Activation Norm")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{key}_layernorms.png"))
    plt.close()
