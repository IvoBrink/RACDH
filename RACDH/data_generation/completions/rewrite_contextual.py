from RACDH.data_generation.entity_recognition.spaCy import get_entities
from RACDH.data_generation.removing.prompt_remove_entity import remove_entity_from_passage
import random
from RACDH.data_generation.utils.print import *

def rewrite_contextual_passage(passage, entity, appending_sentence):
    entities = get_entities(passage)
    
    # Get list of entities different from our target entity
    other_entities = [e for (e, _) in entities if e != entity]
    
    # If no other entities available, return None
    if not other_entities:
        print_warning(f"No other entities found to replace {entity}")
        return None, None
        
    removed_entity = random.choice(other_entities)
    rewritten = remove_entity_from_passage(removed_entity, passage)
    
    if rewritten is not None and entity in rewritten:
        return rewritten + " " + appending_sentence, removed_entity
    else:
        print_warning("Core entity was removed in process, defaulting to standard text")
        return None, None
    
    
    