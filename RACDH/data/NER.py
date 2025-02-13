import torch
import re
from transformers import AutoTokenizer, AutoModelForTokenClassification

# ------------------------------------------------------------------------------
# 0) LOADING MODEL & TOKENIZER
# ------------------------------------------------------------------------------
tokenizer = AutoTokenizer.from_pretrained("dslim/bert-base-NER")
model = AutoModelForTokenClassification.from_pretrained("dslim/bert-base-NER")

sentence = (
    """The history of Dallas, Texas, United States from 1874 to 1929 documents the city's rapid growth and emergence as a major center for transportation, trade and finance. Originally a small community built around agriculture, the convergence of several railroads made the city a strategic location for several expanding industries. During the time, Dallas prospered and grew to become the most populous city in Texas, lavish steel and masonry structures replaced timber constructions, Dallas Zoo, Southern Methodist University, and an airport were established. Conversely, the city suffered multiple setbacks with a recession from a series of failing markets ( the " Panic of 1893 " ) and the disastrous flooding of the Trinity River in the spring of 1908."""
)

sentence = ("""The chronicle of Elyria, Nova Haven, from 1874 to 1929 chronicles the metropolis's swift expansion and ascension as a pivotal hub for navigation, commerce, and economics. Initially a humble settlement centered around husbandry, the convergence of several canals made the city a pivotal location for several burgeoning industries. During this period, Elyria flourished and grew to become the most populous metropolis in the Kingdom of Valtania, opulent crystal and obsidian structures supplanted wooden constructions, the Elyrian Conservatory, the Valtanian Academy of Sciences, and a seaport were""")

# 1) Tokenize
encoding = tokenizer(sentence, return_tensors="pt")

with torch.no_grad():
    outputs = model(**encoding)

logits = outputs.logits  # shape: [batch_size, seq_length, num_labels]
predictions = torch.argmax(logits, dim=-1)[0]  # shape: [seq_length]

id2label = model.config.id2label
predicted_labels = [id2label[idx.item()] for idx in predictions]

tokens = tokenizer.convert_ids_to_tokens(encoding["input_ids"][0])

# ------------------------------------------------------------------------------
# PRINT TOKEN-LEVEL LABELS (DEBUG)
# ------------------------------------------------------------------------------
# print("_" * 50)
# for t, lbl in zip(tokens, predicted_labels):
#     print(f"{t:15} -> {lbl}")
# print("_" * 50)


# ------------------------------------------------------------------------------
# HELPER: Remove "##" prefix
# ------------------------------------------------------------------------------
def clean_subword_token(token: str) -> str:
    return re.sub(r"^##", "", token)

# ------------------------------------------------------------------------------
# (A) IMPROVED UNIFY LABELS ACROSS WORD BOUNDARIES
# ------------------------------------------------------------------------------
def unify_labels_across_word_boundaries(tokens, labels):
    """
    If any sub-token in a word has a non-"O" label, unify the entire word to that label.
    Also, if we discover a 'B-XYZ' label for the first sub-token in the word,
    we convert subsequent sub-tokens in that word to 'I-XYZ' to ensure a single chunk.

    Word boundary logic:
      - A 'word' starts at the first token that does NOT begin with "##"
        (ignoring [CLS], [SEP]).
      - Subsequent tokens that start with "##" are part of that same word.
      - Once we see another token that doesn't start with "##", that's a new word.

    We'll pick the FIRST non-O label we encounter in each word. 
    E.g., if it's "B-PER", the entire word will become B-PER/I-PER. 
    If we later see "I-PER" in the same word, it won't override the earlier B-PER.
    """
    new_labels = labels[:]  # make a copy so we don't modify in-place

    current_word_start = 0
    current_word_label = "O"   # e.g., "B-PER" or "I-PER" or "O"
    current_main_tag = None    # e.g., "PER", "LOC", "ORG", etc.

    def finalize_word(start_idx, end_idx, word_label, main_tag):
        """
        If word_label != "O", unify all tokens in [start_idx, end_idx) to that label.
        If the label starts with "B-XYZ", subsequent tokens are forced to "I-XYZ".
        """
        if word_label == "O":
            return

        if word_label.startswith("B-"):
            # E.g., "B-PER" -> main_tag = "PER"
            pass
        elif word_label.startswith("I-"):
            # E.g., "I-LOC" -> main_tag = "LOC"
            pass
        else:
            # If we see plain "PER" (rare in some label sets), treat it as "B-PER"
            word_label = f"B-{word_label}"

        # Guarantee "B-XYZ" for the first sub-token, then "I-XYZ" for the rest
        first_label = word_label
        for i in range(start_idx, end_idx):
            token = tokens[i]
            if token in ["[CLS]", "[SEP]"]:
                continue

            if i == start_idx:
                new_labels[i] = first_label
            else:
                # Force subsequent tokens to I-XYZ
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

        # If we find a non-"O" label, unify the entire word to that label
        # if we haven't already assigned a label.
        if orig_label != "O" and current_word_label == "O":
            # e.g., "B-LOC" or "I-ORG" ...
            current_word_label = orig_label
            # parse out the main tag from "B-XXX" or "I-XXX"
            if "-" in current_word_label:
                _, tag = current_word_label.split("-", 1)
                current_main_tag = tag
            else:
                # if it's a plain "LOC" (rare), use that
                current_main_tag = current_word_label
                current_word_label = f"B-{current_main_tag}"

    # finalize the last chunk
    finalize_word(current_word_start, len(tokens), current_word_label, current_main_tag)

    return new_labels


# 1) Force partial-word labeling to unify within each single word
predicted_labels = unify_labels_across_word_boundaries(tokens, predicted_labels)

# ------------------------------------------------------------------------------
# (B) YOUR EXISTING MERGE LOGIC FOR B-/I- ENTITIES
# ------------------------------------------------------------------------------
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
            current_entity_text += cleaned_token  # no space
        else:
            current_entity_text += " " + cleaned_token

    else:
        # label == "O" or mismatch
        if current_entity_text:
            final_entities.append((current_entity_text, current_label))
            current_entity_text = ""
            current_label = None

# close out if something remains
if current_entity_text:
    final_entities.append((current_entity_text, current_label))

# ------------------------------------------------------------------------------
# PRINT MERGED ENTITIES
# ------------------------------------------------------------------------------
print("\nFinal Merged Entities:")
for word, entity_label in final_entities:
    print(f"{word} -> {entity_label}")
