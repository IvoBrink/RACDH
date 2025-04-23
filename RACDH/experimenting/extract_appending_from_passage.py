import sys
import os
import json
import spacy
from tqdm import tqdm
sys.path.append(os.path.abspath("/home/ibrink/RACDH/RACDH/"))
from pathlib import Path
from RACDH.data_generation.utils.reading_data import load_json
from RACDH.data_generation.utils.writing_data import write_to_json
from RACDH.config import params

# ── spaCy pipeline ──────────────────────────────────────────────
nlp = spacy.load("en_core_web_sm")           # python -m spacy download en_core_web_sm  (once)

def get_last_sentence(text: str) -> str:
    """Return the last non-empty sentence in `text`."""
    doc = nlp(text)
    sentences = [s.text.strip() for s in doc.sents if s.text.strip()]
    return sentences[-1] if sentences else ""

# ── main ────────────────────────────────────────────────────────
if __name__ == "__main__":
    root = Path(f"{params.target_name}/{params.instruct_name}")
    samples = load_json(str(root / "completions.json"))

    for s in tqdm(samples, desc="Processing samples"):
        s["appending_sentence"] = get_last_sentence(s["passage"])

    write_to_json("completions.json", samples)

    
