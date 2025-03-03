from RACDH.config import params

COLOR_CODES = {
    "RED": "\x1b[31m",   # Red for Person
    "BLUE":    "\x1b[34m",   # Blue for Organization
    "YELLOW":   "\x1b[33m",   # Yellow for Date/Time
    "GREEN":    "\x1b[32m",   # Green for Geo-Political Entities (like countries, cities)
    "MAGENTA":   "\x1b[35m",   # Magenta for Nationalities or Groups
    "CYAN":   "\x1b[36m",   # Cyan for works of art (books, shows, etc.)
}
RESET_CODE = "\x1b[0m"      # Reset color code to default


def highlight_entities(text, entities):
    """
    Given the full text and a list of (entity_text, start_offset),
    return a new string that highlights recognized entities.
    """
    # 1) Sort entities by their start offset
    entities_sorted = sorted(entities, key=lambda x: x[1])

    color_dict = {
        "Correct" : COLOR_CODES["GREEN"],
        "Redacted due to title overlap" : COLOR_CODES["RED"],
        "Redacted too frequent" : COLOR_CODES["YELLOW"]
    }
    
    highlighted_text_parts = []
    last_index = 0
    
    # 2) Build the highlighted text
    for entity in entities_sorted:
        if len(entity) == 2:
           entity_str, start_offset = entity
           category = None
        else:
            entity_str, start_offset, category = entity
        # Append any text before this entity
        highlighted_text_parts.append(text[last_index:start_offset])
        
        # Append the entity in a highlighted form
        # For example, use double-asterisks around the entity text
        if params.print_entity_categories:
            highlighted_text_parts.append(f"{color_dict.get(category, COLOR_CODES['GREEN'])}**{entity_str}({category})**{RESET_CODE}")
        else:
            highlighted_text_parts.append(f"{color_dict.get(category, COLOR_CODES['GREEN'])}{entity_str}{RESET_CODE}")

        
        # Move the pointer to the end of this entity in the original text
        # (If the length in the text matches entity_str exactly)
        last_index = start_offset + len(entity_str)
        
    # 3) Append any remaining text after the final entity
    highlighted_text_parts.append(text[last_index:])
    
    # 4) Join all parts
    return "".join(highlighted_text_parts)