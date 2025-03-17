from RACDH.data_generation.target_model import generate_completion
from RACDH.config import params
from RACDH.data_generation.utils.print import *
from RACDH.data_generation.cross_encoder import get_similarity_score

def evaluate_knowledge(tests, entity):
    tests_passed = 0
    for test in tests:
        input,output = generate_completion(test, entity, max_new_tokens=32, temperature=0.5, debug=params.debug)

        if knows(output, entity):
            tests_passed += 1
            if params.debug: print("Test passed")
        else:
            if params.debug: print("Test failed")

    verdict = "KNOWS" if tests_passed >= params.knowledge_tests_threshold else "IGNORANCE"
    if params.debug: print_h2(f"{entity} passed {tests_passed}/{len(tests)} tests. {verdict}")
    return tests_passed


def knows(output, entity):
    if entity.lower() in output.lower():
        return True
    score = get_similarity_score(output, entity)
    if params.debug: print_h4(f"Similarity score = {score}")
    return entity.lower() in output.lower() or score >= params.similarity_threshold_entity
