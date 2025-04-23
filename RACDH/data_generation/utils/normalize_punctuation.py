from RACDH.data_generation.instruct_model import generate_completion_GPT


def normalize_punctuation(text: str) -> str:
    prompt = f"""You are given a text that contains inconsistent spacing. You must produce a version of the text with only the spacing corrected.

Here are your rules:

1. Do NOT remove or change any words or punctuation symbols. Only adjust spaces.
2. Remove extra spaces immediately after opening quotes, parentheses, or brackets.
   - e.g., `"   Einstein` should become `"Einstein`.
3. Remove extra spaces immediately before closing quotes, parentheses, or brackets.
   - e.g., `Contigo   ")` should become `Contigo")`.
4. Remove extra spaces around apostrophes for contractions or possessives.
   - e.g., `album 's` → `album's`.
5. Remove extra spaces before punctuation marks (commas, periods, colons, semicolons) if they appear.
   - e.g., `song  ,` → `song,`.
6. Do NOT introduce new punctuation; only fix the spacing.
7. Maintain all words and their order exactly as given.

Now, apply these rules to the following text and return the corrected version:

Text with inconsistent spacing:
<<< {text} >>>

Text with consistent spacing (end with >>>):
<<<"""
    text = generate_completion_GPT(prompt)
    return text
