from RACDH.data_generation.target_model import generate_completion
from RACDH.config import params
from RACDH.data_generation.utils.print import *

def evaluate_knowledge(tests, entity):
    tests_passed = 0
    for test in tests:
        input,output = generate_completion(test, max_new_tokens=32, temperature=0.5, debug=params.debug)
        if knows(output, entity):
            tests_passed += 1
            if params.debug: print("Test passed")
        else:
            if params.debug: print("Test failed")

    verdict = "KNOWS" if tests_passed >= params.knowledge_tests_threshold else "IGNORANCE"
    if params.debug: print_h2(f"{entity} passed {tests_passed}/{len(tests)} tests. {verdict}")
    return tests_passed >= params.knowledge_tests_threshold


def knows(output, entity):
    #TODO: be more thorough here. Maybe also check for similar mentions.
    return entity.lower() in output.lower()
