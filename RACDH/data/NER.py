import torch
import re
from transformers import AutoTokenizer, AutoModelForTokenClassification

# ------------------------------------------------------------------------
# LOAD THE MODEL & TOKENIZER (done once, globally)
# ------------------------------------------------------------------------
tokenizer = AutoTokenizer.from_pretrained("dslim/bert-base-NER")
model = AutoModelForTokenClassification.from_pretrained("dslim/bert-base-NER")

# ------------------------------------------------------------------------
# (OPTIONAL) HELPER FUNCTIONS
# ------------------------------------------------------------------------

def clean_subword_token(token: str) -> str:
    """
    Remove the '##' prefix from subword tokens.
    """
    return re.sub(r"^##", "", token)

def unify_labels_across_word_boundaries(tokens, labels):
    """
    If any sub-token in a word has a non-"O" label, unify the entire word to that label.
    Also, ensure consistent B- and I- prefixes within the same word.
    """
    new_labels = labels[:]  # copy so we don't modify in-place

    current_word_start = 0
    current_word_label = "O"
    current_main_tag = None

    def finalize_word(start_idx, end_idx, word_label, main_tag):
        """
        For tokens in [start_idx, end_idx), unify them under one label if word_label != "O".
        """
        if word_label == "O":
            return

        # Ensure the first sub-token is B-XYZ and subsequent sub-tokens are I-XYZ
        if not (word_label.startswith("B-") or word_label.startswith("I-")):
            word_label = f"B-{word_label}"

        # Extract main tag from "B-LOC", "I-ORG", etc.
        if "-" in word_label:
            _, main_tag = word_label.split("-", 1)

        # Mark the range
        for i in range(start_idx, end_idx):
            token = tokens[i]
            if token in ["[CLS]", "[SEP]"]:
                continue

            if i == start_idx:
                new_labels[i] = f"B-{main_tag}"
            else:
                new_labels[i] = f"I-{main_tag}"

    for i, (token, orig_label) in enumerate(zip(tokens, labels)):
        if token in ["[CLS]", "[SEP]"]:
            continue  # skip special tokens

        is_new_word = not token.startswith("##")

        if is_new_word:
            # finalize the previous word chunk
            finalize_word(current_word_start, i, current_word_label, current_main_tag)

            # start a new chunk
            current_word_start = i
            current_word_label = "O"
            current_main_tag = None

        # If current sub-token has a non-"O" label, assign it if we haven't done so already
        if orig_label != "O" and current_word_label == "O":
            current_word_label = orig_label
            if "-" in current_word_label:
                _, current_main_tag = current_word_label.split("-", 1)
            else:
                # if it's a plain label (rare), treat it as B-label
                current_main_tag = current_word_label
                current_word_label = f"B-{current_main_tag}"

    # finalize the last chunk
    finalize_word(current_word_start, len(tokens), current_word_label, current_main_tag)

    return new_labels

# ------------------------------------------------------------------------
# MAIN FUNCTION: NER
# ------------------------------------------------------------------------
def NER(sentence: str):
    """
    Takes a raw sentence (string) and returns a list of (word, entity_label)
    pairs, using the dslim/bert-base-NER model.
    """
    # 1) Tokenize
    encoding = tokenizer(sentence, return_tensors="pt")

    # 2) Forward pass
    with torch.no_grad():
        outputs = model(**encoding)

    # 3) Get predictions
    logits = outputs.logits  # shape: [batch_size, seq_length, num_labels]
    predictions = torch.argmax(logits, dim=-1)[0]  # shape: [seq_length]

    id2label = model.config.id2label
    predicted_labels = [id2label[idx.item()] for idx in predictions]
    tokens = tokenizer.convert_ids_to_tokens(encoding["input_ids"][0])

    # 4) Unify labels across sub-word boundaries
    predicted_labels = unify_labels_across_word_boundaries(tokens, predicted_labels)

    # 5) Merge tokens by B-/I- entity logic
    final_entities = []
    current_entity_text = ""
    current_label = None

    for token, label in zip(tokens, predicted_labels):
        if token in ["[CLS]", "[SEP]"]:
            continue

        if label.startswith("B-"):
            # If we were building an entity, close it
            if current_entity_text:
                final_entities.append((current_entity_text, current_label))
                current_entity_text = ""

            current_label = label[2:]  # e.g., "PER" from "B-PER"
            cleaned_token = clean_subword_token(token)
            current_entity_text = cleaned_token

        elif label.startswith("I-") and current_label == label[2:]:
            cleaned_token = clean_subword_token(token)
            if token.startswith("##"):
                # same word piece, no space
                current_entity_text += cleaned_token
            else:
                current_entity_text += " " + cleaned_token
        else:
            # label == "O" or mismatch
            if current_entity_text:
                final_entities.append((current_entity_text, current_label))
                current_entity_text = ""
                current_label = None

    # Close out any remaining entity
    if current_entity_text:
        final_entities.append((current_entity_text, current_label))

    return final_entities

# ------------------------------------------------------------------------
# EXAMPLE USAGE:
# (Uncomment to test)
# ------------------------------------------------------------------------
# if __name__ == "__main__":
#     text = "Barack Obama was born in Hawaii. He was elected president in 2008."
#     entities = NER(text)
#     for word, label in entities:
#         print(f"{word} -> {label}")
