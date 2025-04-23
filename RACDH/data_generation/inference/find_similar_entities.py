from RACDH.data_generation.entity_recognition.spaCy import get_entities
from RACDH.data_generation.cross_encoder import get_similarity_score
from RACDH.data_generation.utils.print import print_warning
from RACDH.config import params

def find_similar_entities(text, entity):
    # do entity recognition
    entities = get_entities(text)
    print(f"Entities found: ")
    # check similarity across list
    for found_entity_tuple in entities:
        # Extract the entity text from the tuple (first element)
        found_entity_text = found_entity_tuple[0]
        score = get_similarity_score(found_entity_text, entity)
        print(f"Score for {found_entity_text} : {score}")

        if score > params.similarity_threshold_entity:
            print_warning(f"\n\n{text} \n\n{entity}\n\n{found_entity_text}\n\n")
            return found_entity_text
        
    return None
    # return if one above threshold