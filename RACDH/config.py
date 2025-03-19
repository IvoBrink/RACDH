

class params:
    wiki_path = "/home/ibrink/RACDH/RACDH/MIND/auto-labeled/wiki/"
    output_path = "/home/ibrink/RACDH/RACDH/RACDH/data/"

    taget_model_name_or_path = "meta-llama/Llama-3.1-8B"
    instruct_model_name_or_path = "meta-llama/Llama-3.1-8B-Instruct"
    openAI_model = "gpt-4o-mini"
    openAI = False


    debug = True
    print_entity_categories = True

    knowledge_tests_threshold = 1

    similarity_threshold_entity = 0.6

        ## Naming models
    target_name = taget_model_name_or_path.split('/')[-1]
    instruct_name = openAI_model if openAI else instruct_model_name_or_path.split('/')[-1]