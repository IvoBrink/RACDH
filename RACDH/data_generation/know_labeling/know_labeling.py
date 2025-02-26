import sys
import os
sys.path.append(os.path.abspath("/home/ibrink/RACDH/RACDH/"))
from RACDH.data_generation.know_labeling.design_completions.all_completion_types import generate_one_for_all_types



if __name__ == "__main__":
    passage = """William John " Willie " Irvine ( born 18 June 1943 ) is a former professional footballer who played as a centre forward . Born in Eden , County Antrim , into a large family , he grew up in the nearby town of Carrickfergus . He did well at school , but chose to pursue a career in professional football and initially played for local club Linfield . After a spell in amateur football , Irvine travelled to England for a trial with Burnley at the age of 16 . He was offered a professional deal and spent three years playing for the youth and reserve teams , before making his senior debut at the end of the 1962 – 63 season . Over the following seasons , Irvine became a regular feature of the Burnley team and in the 1965 – 66 campaign , he scored 29 goals and was the highest goalscorer in the Football League First Division ."""
    entity = "Linfield"
    generate_one_for_all_types(passage, entity)
    # Load in the passages and entities
    # For each passage
        # For each entity in passage
            # Generate akice-bob, truncate and question completions
            # Have target model answer them
            # Compare answer to correct
            # Label if model knows