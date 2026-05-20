import os as _os

class params:
    wiki_path = _os.environ.get("WIKI_PATH", _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "MIND", "auto-labeled", "wiki") + "/")
    output_path = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "data") + "/"

    # target_model_name_or_path = "meta-llama/Llama-3.1-8B"
    # target_model_name_or_path = "mistralai/Mistral-7B-v0.1"
    target_model_name_or_path = "Qwen/Qwen2.5-7B"
    instruct_model_name_or_path = "meta-llama/Llama-3.1-8B-Instruct"
    openAI_model = "gpt-4o-mini"
    openAI = True


    debug = True
    print_entity_categories = True

    knowledge_tests_threshold = 1

    similarity_threshold_entity = 0.6
    similarity_check_inference = True

    ## Naming models
    target_name = target_model_name_or_path.split('/')[-1]
    instruct_name = openAI_model if openAI else instruct_model_name_or_path.split('/')[-1]


    # Bias checking
    last_sentence = False

