import spacy
import re

nlp = spacy.load('en_core_web_trf')

def clean_trailing_punctuation(entity):
    # Remove leading and trailing punctuation
    while entity and entity[0] in ',:.;\'?!"()\'':
        entity = entity[1:]
    while entity and entity[-1] in ',:;.\'?!"()\'':
        entity = entity[:-1]
    return entity.strip()

def find_boundaries(text, words):
    boundaries = []
    for word in words:
        start = 0
        ntext = text
        while True:
            start = ntext.find(word)
            if start == -1:
                break
            end = start + len(word) - 1
            while start > 0 and ntext[start-1] != " ":
                start -= 1
            while end < len(ntext)-1 and ntext[end+1] != " ":
                end += 1
            entity = "".join([ntext[i] for i in range(start, end+1)])
            entity = clean_trailing_punctuation(entity)
            if entity:  # Only add if entity is not empty after cleaning
                boundaries.append(entity)
            ntext = ntext[end+1:]
    return boundaries


def delete_substrings(lst):
    substrings = []
    lst = list(set(lst))
    for s in lst:
        if any(s in o for o in lst if o != s):
            substrings.append(s)
    for s in substrings:
        lst.remove(s)
    return lst


def get_entities(text):
    entities_ = list(set([str(e) for e in nlp(text).ents]))
    entities_ = find_boundaries(text, entities_)
    entities = delete_substrings(entities_)
    all_entities = []
    for i in range(len(text)):
        for e in entities:
            if text[i:].startswith(e):
                all_entities.append((e, i))
    return all_entities