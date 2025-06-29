import os
import json
from tqdm import tqdm  # Added tqdm



def load_squad(path):
    raw = json.load(open(path, encoding="utf-8"))
    for art in raw.get("data", raw):
        for para in art.get("paragraphs", [art]):
            ctx = para["context"]
            for qa in para["qas"]:
                if qa.get("is_impossible"):
                    continue
                answers = list({a["text"] for a in qa["answers"]})  # remove duplicates
                yield {"question": qa["question"], "answers": answers, "context": ctx}


if __name__ == "__main__":
    # Load all samples into a list so we can iterate multiple times
    squad_samples = list(load_squad("RACDH/data/squad/dev-v2.0.json"))
    # I want to check for each answer how many times it occurred in another context
    from collections import defaultdict
    answer_counts = defaultdict(int)
    # For each sample, for each answer, check if it appears in another context
    for i, sample in enumerate(tqdm(squad_samples, desc="Samples")):
        for answer in sample["answers"]:
            answer_lower = answer.lower()
            found_in_another_context = False
            for j, other_sample in enumerate(squad_samples):
                if i == j:
                    continue
                # Check if answer appears in the context of another sample,
                # but is not one of that sample's answers
                if answer_lower in other_sample["context"].lower() and all(answer_lower != a.lower() for a in other_sample["answers"]):
                    answer_counts[answer] += 1
                    found_in_another_context = True
                    break  # Only count once per answer per sample
    # top 10 answers
    print(sorted(answer_counts.items(), key=lambda x: x[1], reverse=True)[:100])
