import sys
import os
sys.path.append(os.path.abspath("/home/ibrink/RACDH/RACDH/"))
from RACDH.data_generation.know_labeling.design_completions.all_completion_types import generate_one_for_all_types
from RACDH.data_generation.know_labeling.generate_completions.target_completion_knowing import target_completion



if __name__ == "__main__":
    passage = """Clive Andrew Mantle ( born 3 June 1957 ) is an English actor . He is best known for playing general surgeon Dr Mike Barrett in the BBC hospital drama series Casualty and Holby City in the 1990s , and is also noted for his role as Little John in the cult 1980s fantasy series Robin of Sherwood .."""
    entity = "Mike Barett"
    completions = generate_one_for_all_types(passage, entity)
    target_completion(completions[0], entity)

    # Load in the passages and entities
    # For each passage
        # For each entity in passage
            # Generate akice-bob, truncate and question completions
            # Have target model answer them
            # Compare answer to correct
            # Label if model knows