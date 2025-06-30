# RACDH: Real-time Attribution for Context vs. Parametric Knowledge in LLMs

This repo implements a lightweight, probing-based framework to detect where large language models source their generated content: from the prompt context or from internal weights. It combines self-supervised data generation, hidden-state classifiers, and attribution-driven hallucination signals.

## Key features

* **AttriWiki**: automatic dataset builder that forces LLMs to retrieve withheld entities from either context or parametric memory.
* **Real-time attribution classifier**: a compact linear probe on decoder hidden states reaches up to 96% Macro-F1 on LLaMA-3.1-8B and Mistral-7B, generalizing to SQuAD and WebQuestions without retraining.
* **Correlation with hallucinations**: reveals that attribution mismatches increase wrong-answer odds by \~70%.
* **Fast + interpretable**: no extra forward/backward passes, operates in real time at token level.

## Installation



## Papers & citation

This project is based on the MSc thesis:

> Ivo Brink (2025). *Real-time Knowledge Attribution as an Early-Warning Signal for LLM Hallucinations*. University of Amsterdam. [PDF](link-if-available)

If you use this work, please cite the thesis.
