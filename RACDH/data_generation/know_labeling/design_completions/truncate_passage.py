import nltk
# Make sure NLTK's "punkt" is downloaded:
# nltk.download("punkt")
from nltk.tokenize import sent_tokenize
from RACDH.data_generation.utils.print import *

def truncate_passage_at_entity(passage: str, entity: str):
    """
    Splits the passage into sentences using NLTK's sent_tokenize,
    then checks each sentence in order.

    1) If (ignoring leading spaces) the sentence starts with the entity, return None.
    2) If the entity appears elsewhere in that sentence, return the passage truncated
       right before that occurrence.
    3) If the entity is never found, print a warning and return None.

    Additionally:
    - If the entity appears more than once in the passage, print a warning (should not happen).
    """

    # First, check if the entity occurs more than once:
    if params.debug: print_h3("Truncate passage 'till entity")
    entity_count = passage.count(entity)
    if entity_count > 1:
        print_warning(
            f"Truncate Passage: '{entity}' occurs {entity_count} times in the text. "
            f"This should not be possible."
        )

    # Use NLTK's sentence tokenizer for a more robust split.
    sentences = sent_tokenize(passage)

    # Track the position in 'passage' to map each sentence back to its absolute index.
    position = 0
    passage_length = len(passage)

    for sentence in sentences:
        # Find where this sentence appears in the original passage
        idx = passage.find(sentence, position)
        if idx == -1:
            # If we can't find the exact substring from 'position' onward, move on.
            continue

        sentence_start = idx

        # Check if sentence (ignoring leading whitespace) starts with the entity
        if sentence.lstrip().startswith(entity):
            print_warning("Within passage sentence is started with entity.")
            return None

        # Otherwise, see if the entity occurs within this sentence
        index_in_sentence = sentence.find(entity)
        if index_in_sentence != -1:
            # Found the entity inside the sentence
            absolute_index = sentence_start + index_in_sentence
            if params.debug:
                print(passage[:absolute_index])
            return passage[:absolute_index]

        # Move 'position' to the end of this sentence to continue
        position = sentence_start + len(sentence)
        if position >= passage_length:
            break

    # If we finish the loop without returning, the entity was never found.
    print_warning(f"Truncate Passage: '{entity}' was not found in the given passage.")
    return None
