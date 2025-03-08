from RACDH.data_generation.know_labeling.design_completions.bob_alice import generate_alice_bob_example

def generate_one_for_all_types(passage, entity):
    ex1 = generate_alice_bob_example(passage, entity)
    print(ex1)
    return [ex1]
    # generate alice-bob
    # generate truncate
    # generate question


def all_completion_types(passage, entities):
    entity_completions = {}
    for entity in entities:
        entity_completions[entity] = generate_one_for_all_types(passage, entity)
    return entity_completions
