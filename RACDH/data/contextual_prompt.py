

example_original = (
    "Original example passage:\n"
    "\"Neil Armstrong was an American astronaut and the first person to walk on the Moon. He was born on August 5, 1930, in Wapakoneta, Ohio, and joined NASA in 1962.\""
)
example_rewritten = (
    "Example rewritten passage:\n"
    "\"Nyra Silverwind was a renowned astral explorer and the first individual to step onto the Celestial Reef. She emerged on Nightfall 3, 1930, in the city-state of Avaria, and enlisted in the Radiant Fleet Academy in 1962.\""
)
        
user_prompt = f"""{example_original}

{example_rewritten}

Notice how the new text:
- Maintains a similar sentence structure.
- Replaces all real names, dates, places, and organizations with entirely made-up ones.
- Keeps an encyclopedic tone but no verifiable real facts remain.

Now It is Your Turn

Below is your **real** Wikipedia passage. Transform it in the same style: 
1. Keep the paragraph count and approximate sentence structure.  
2. Replace names, locations, dates, and other facts with imaginary ones.  
3. Output only your final fictional rewrite.  
4. Do not add extra commentary or disclaimers.


Original passage:
{sentences}
Example rewritten passage:
"""