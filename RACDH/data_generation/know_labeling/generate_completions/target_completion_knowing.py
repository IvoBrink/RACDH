from RACDH.data_generation.target_model import generate_completion

def target_completion(sentence, entity):
    completion = generate_completion(sentence, max_new_tokens=32, temperature=0.5, debug=True)