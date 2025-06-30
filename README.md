# RACDH: Real-time Attribution for Context vs. Parametric Knowledge in LLMs

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
git clone https://github.com/IvoBrink/RACDH.git
cd RACDH
```

### Option 1: Using Conda (recommended)
If you use [conda](https://docs.conda.io/), you can create an environment with all dependencies:

```bash
conda env create -f RACDH/env.yml
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

## Papers & citation

This project is based on the MSc thesis:

> Ivo Brink (2025). *Real-time Knowledge Attribution as an Early-Warning Signal for LLM Hallucinations*. University of Amsterdam.

If you use this work, please cite the thesis.
