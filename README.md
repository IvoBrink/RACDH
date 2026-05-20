# RACDH: Real-time Attribution Classification to Detect Hallucinations.

This repo implements a lightweight, probing-based framework to detect where large language models source their generated content: from the prompt context or from internal weights. It combines self-supervised data generation, hidden-state classifiers, and attribution-driven hallucination signals.

## Key features

* **`data_generation/`**: Automatic dataset builder ("AttriWiki") that forces LLMs to retrieve withheld entities from either context or parametric memory.
* **`classification/`**: Real-time attribution classifier—a compact linear probe on decoder hidden states that achieves up to 96% Macro-F1 on LLaMA-3.1-8B and Mistral-7B, and generalizes to SQuAD and WebQuestions without retraining.
* **`analysis/`**: Correlation analysis showing that attribution mismatches increase wrong-answer odds by ~70%.
* **Fast & interpretable**: No extra forward/backward passes; operates in real time at the token level.

## Installation
To get started, clone this repository and install the required dependencies.

First, clone the repository:

```bash
git clone https://github.com/[anonymous]/RACDH.git
cd RACDH
```

### Option 1: Using Conda (recommended)
If you use [conda](https://docs.conda.io/), you can create an environment with all dependencies:

```bash
conda env create -f environment.yml
conda activate RACDH
```

### Option 2: Using pip
If you prefer pip, install the dependencies from `requirements.txt`:

```bash
pip install -r requirements.txt
```

> **Note:** The `requirements.txt` includes both conda and pip-style dependencies. If you encounter issues, prefer the conda environment or manually install any missing packages.

## Workflow: Running the Pipeline

Below is the recommended order for running the main components of this project:

### 1. Data Generation (5 steps)
The data generation pipeline creates the datasets for probing LLM knowledge sources. **Run these scripts in the following order:**

1. **Entity Extraction**: Extract entities from Wikipedia or your corpus.
   - `python RACDH/data_generation/entity_recognition/extract_entities.py`
2. **Know Labeling**: Determine which entities a model already knows (before generating completions).
   - `python RACDH/data_generation/know_labeling/know_labeling.py`
3. **Removal**: Remove known entities from the dataset.
   - `python RACDH/data_generation/removing/remove_known_entities.py`
4. **Completions Generation**: Generate LLM completions for the entities.
   - `python RACDH/data_generation/completions/add_completions.py`
5. **Hidden State Extraction**: Extract hidden states from the LLM for each completion.
   - `python RACDH/data_generation/inference/extract_hiddens.py`

**Optional: Bias Checker**
- You can optionally run the bias checker to analyze potential biases in the generated data or model predictions.
  - Example: `python RACDH/bias_checker/text_classification.py`

### 2. Classification (Model Training)
Train a classifier to attribute knowledge source using the generated data:
- Example: `python RACDH/classification/all_layer_linear.py` (this is the best model, params are already optimal)

### 3. Validation (Out-of-Domain Generalization)
Test the classifier on new datasets:
1. **Sample Generation**: Create validation samples from out-of-domain datasets.
   - `python RACDH/classification/datasets/samples.py`
2. **Validation**: Run the trained classifier on these samples.
   - `python RACDH/classification/datasets/validate.py`

### 4. Analysis (Correlation with Hallucination)
Analyze how attribution mismatches correlate with hallucination:
- Example: `python RACDH/analysis/analysis_hallucination.py`

> See the respective script files for more details and arguments. Outputs are saved in the `RACDH/data/` directory.

**Data availability:**
- Some data and results are already included in this repository. Directories are organized by target model (e.g., `Llama-3.1-8B`, `Mistral-7B-v0.1`).
- **Hidden states are not included** due to their large size. To obtain them, contact the author or recreate them using the provided scripts.

---

**Configuration:**
- For major changes that affect multiple scripts (e.g., model selection, data paths), edit the central configuration file: `RACDH/config.py`.
- For changes specific to a single script (e.g., input/output files, batch size), use the command-line arguments provided by that script (see `--help` for options).

## Reference: Tighidet et al. Knowledge-Probing Framework

The `tighidet/` directory contains a reference implementation of the knowledge-probing framework from the EMNLP BlackboxNLP 2024 paper:

> Zineddine Tighidet, Andrea Mogini, Jiali Mei, Benjamin Piwowarski, Patrick Gallinari (2024). *Probing Language Models on Their Knowledge Source*. arXiv:2410.05817.

This framework probes LLMs on the conflict between **parametric knowledge (PK)** (stored in model weights) and **contextual knowledge (CK)** (provided at inference time) using controlled counter-factual prompts. It trains per-layer linear classifiers via leave-one-relation-group-out cross-validation to predict the knowledge source from hidden-state activations.

**It is not part of the main RACDH pipeline**—it is included for direct comparison. The RACDH approach differs in that it targets real-time hallucination detection from a self-supervised dataset (AttriWiki), while Tighidet et al. focus on controlled knowledge-conflict scenarios using the ParaRel dataset.

The original Tighidet code was extended to load the RACDH probe directly into their evaluation framework:
- `attriwiki_probe/` was added, containing the `WeightedAggLogReg` model definition and pre-trained `.joblib` probes (Llama-3.1-8B, Mistral-7B-v0.1, Qwen2.5-7B) trained on the AttriWiki dataset.
- `scripts/main.py` was extended with a step 7 that runs the RACDH probe on the Tighidet counter-parametric knowledge scenarios, producing `WeightedAgg` metrics alongside the original per-layer results.
- `scripts/bootstrap_results.py` was added to compare macro-F1 ± std of the Tighidet best-layer classifier against the RACDH probe on the same data.

Key differences:

| | RACDH | Tighidet et al. |
|---|---|---|
| Dataset | AttriWiki (self-supervised) | ParaRel (reformatted) |
| Prompt design | Withheld entities from Wikipedia | Counter-factual knowledge conflicts |
| Classifier | Weighted-aggregation logistic regression | Per-layer logistic regression |
| Token positions | First/last entity & generation tokens | relation_query, object, subject_query, first |
| Goal | Real-time hallucination detection | Understanding knowledge source selection |

To run the Tighidet framework independently, see `tighidet/knowledge-probing-framework/README.md`.

---

## Papers & citation

This project is based on the MSc thesis:

> Anonymous (2025). *Probing for Knowledge Attribution in Large Language Models*.

If you use this work, please cite the thesis.

## Codebase Summary

- Total *.py files: 53
- Total code lines: 3381
- Total comment lines: 788
- Total empty lines: 1104
