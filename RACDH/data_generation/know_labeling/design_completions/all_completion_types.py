from RACDH.data_generation.know_labeling.design_completions.bob_alice import generate_alice_bob_example
from RACDH.data_generation.know_labeling.design_completions.truncate_passage import truncate_passage_at_entity
from RACDH.data_generation.know_labeling.design_completions.question import question_example
from RACDH.config import params

def generate_all_knowledge_tests(passage, entity):
    ex1 = generate_alice_bob_example(passage, entity)
    ex2 = truncate_passage_at_entity(passage, entity)
    ex3 = question_example(passage, entity) 

    return [x for x in [ex1, ex2, ex3] if x is not None]
