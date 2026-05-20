import pandas as pd
import re
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Any
from tqdm import tqdm

from src.store_activations import forward_pass_with_hooks


def get_tokens_positions(tokenizer: AutoTokenizer, input_context: str, input_query: str, subject_query: str) -> tuple:
    """returns the indexes of the last token for the context object, query subject, and query relation

    Args:
        tokenizer (AutoTokenizer): the tokenizer of the LLM
        input_context (str): the context just before the query which represents a statement about a subject, relation, and a counter-knowledge object
        input_query (str): the query that comes after the context
        subject_query (str): the subject in the query (which is also the same as the one in the context)

    Returns:
        Tuple: a triplet with the positions of the context object, query subject, and query relation respectively
    """
    object_context_idx = len(tokenizer.encode(f"{input_context}")) - 2  # -2 to not consider the point after the context
    subject_query_idx = len(tokenizer.encode(f"{input_context} {subject_query}")) - 1
    relation_query_idx = len(tokenizer.encode(f"{input_context} {input_query}")) - 1

    return object_context_idx, subject_query_idx, relation_query_idx


def _debug_token_positions(tokenizer: AutoTokenizer, input_context: str, input_query: str, subject_query: str) -> None:
    full_prompt = f"{input_context} {input_query}"
    tokens = tokenizer.encode(full_prompt)
    decoded = [tokenizer.decode([t]) for t in tokens]

    object_context_idx, subject_query_idx, relation_query_idx = get_tokens_positions(
        tokenizer, input_context, input_query, subject_query
    )

    print("\n" + "=" * 70)
    print("TOKEN POSITION DEBUG (first example)")
    print("=" * 70)
    print(f"  BOS token id : {tokenizer.bos_token_id}  |  token[0] id: {tokens[0]}  |  adds BOS: {tokens[0] == tokenizer.bos_token_id}")
    print(f"  Full prompt  : {full_prompt!r}")
    print(f"  Total tokens : {len(tokens)}")
    print(f"  object_context_idx  = {object_context_idx}  → token: {decoded[object_context_idx]!r}")
    print(f"  subject_query_idx   = {subject_query_idx}   → token: {decoded[subject_query_idx]!r}  (expected subject: {subject_query!r})")
    print(f"  relation_query_idx  = {relation_query_idx}  → token: {decoded[relation_query_idx]!r}")
    print("  All tokens:")
    for i, (tid, tok) in enumerate(zip(tokens, decoded)):
        marker = ""
        if i == object_context_idx:  marker = " ← object_context"
        if i == subject_query_idx:   marker = " ← subject_query"
        if i == relation_query_idx:  marker = " ← relation_query"
        print(f"    [{i:3d}] {tid:>8}  {tok!r}{marker}")
    print("=" * 70 + "\n")


def build_prompt_for_knowledge_probing(input_context: str, input_query: str):
    return f"{input_context} {input_query}"


def identify_knowledge_source(
    parametric_knowledge_object: str, counter_knowledge_object: str, output_object: str
) -> str:
    parametric_knowledge_object = parametric_knowledge_object.lower().strip()
    output_object = output_object.lower().strip()
    counter_knowledge_object = counter_knowledge_object.lower().strip()

    # counter_knowledge_object_words = (
    #     counter_param_knowledge_record
    #     .counter_knowledge_object
    #     .lower()
    #     .strip()
    #     .split(" ")
    # )

    return (
        "CK"
        if " ".join(output_object.split(" ")[: len(counter_knowledge_object.split(" "))]) == counter_knowledge_object
        else (
            "PK"
            if " ".join(output_object.split(" ")[: len(parametric_knowledge_object.split(" "))])
            == parametric_knowledge_object
            else "ND"
        )
    )


def _print_examples(df: pd.DataFrame, n: int = 10) -> None:
    sample = df.sample(n=min(n, len(df)), random_state=42)
    print("\n" + "=" * 70)
    print(f"SAMPLE EXAMPLES (n={len(sample)})")
    print("=" * 70)
    for i, (_, row) in enumerate(sample.iterrows(), 1):
        print(f"\n--- Example {i} ---")
        print(f"  Context : {row['prompt_context']}")
        print(f"  Query   : {row['prompt_query']}")
        print(f"  Output  : {row['output_txt'].strip()!r}")
        print(f"  Label   : {row['knowledge_source']}  "
              f"(PK={row['parametric_knowledge_object']}  |  CK={row['counter_knowledge_object']})")
    print("\n" + "=" * 70 + "\n")


def prompt_model_and_store_activations(
    counter_parametric_knowledge_dataset: pd.DataFrame,
    device: str,
    tokenizer: AutoTokenizer,
    model_name: str,
    model: AutoModelForCausalLM,
    include_first_token: bool,
    include_object_context_token: bool,
    include_subject_query_token: bool,
    include_relation_query_token: bool,
    include_mlps: bool = True,
    include_mlps_l1: bool = False,
    include_mhsa: bool = False,
    vertical: bool = True,
    max_gen_tokens: str = 10,
    split_regexp: str = r'"|\.|\n|\\n|,|\(|\<s\>',
) -> tuple[dict[str, list[Any]], dict[str, list[Any]]]:
    """
    Prompts the LLM to determine the knowledge source and saves the activations of specific modules (MLP, MLP-L1, MHSA).

    Args:
        parametric_knowledge (pd.DataFrame): Dataframe containing the parametric knowledge dataset.
        counter_parametric_knowledge_dataset (pd.DataFrame): Dataframe containing the counter-parametric-knowledge dataset.
        device (str): Device where inference on the LLM will be run.
        tokenizer (AutoTokenizer): Tokenizer for the LLM.
        model_name (str): Name of the model (mainly for printing and path reference purposes).
        model (AutoModelForCausalLM): Loaded LLM.
        include_first_token (bool): whether to probing on the first token or not (see control experiment of the paper)
        include_object_context_token (bool): whether to probing on the object token in the context or not
        include_subject_query_token (bool): whether to probing on the object token in the query or not
        include_relation_query_token (bool): whether to probing on the relation token in the query or not
        include_mlps (bool): Whether to include the second layer of the MLP (output) in the activations.
        include_mlps_l1 (bool): Whether to include the first layer of the MLP.
        include_mhsa (bool): Whether to include the activations of the output of the MHSA.
        vertical (bool): Whether the token positions are vertical.
        split_regexp (str): the regexp pattern to parse the output of the LLM after prompting

    Returns:
        prompting_results: A DataFrame containing various results and metrics of the probe as well as the activations
    """

    prompting_results = []
    generation_hidden_states_list = []

    activations = {"mlp": [], "mlp-l1": [], "mhsa": []}

    # We copy to avoid modifying inplace
    counter_parametric_knowledge_dataset = counter_parametric_knowledge_dataset.copy()

    _debug_first = True  # print token-position diagnostics for the first example only

    for _, counter_param_knowledge_record in tqdm(
        counter_parametric_knowledge_dataset.iterrows(),
        total=counter_parametric_knowledge_dataset.shape[0],
        desc=f"Probing {model_name} with prompts...",
    ):
        # Example values:
        # - statement: "Paris is the capital of France"
        # - statement_object: "France"
        # - counter_knowledge_object: "Italy"
        # - statement_query: "Paris is the capital of"
        knowledge_probing_prompt_context = counter_param_knowledge_record.statement.replace(
            counter_param_knowledge_record.statement_object, counter_param_knowledge_record.counter_knowledge_object
        )
        knowledge_probing_prompt_query = counter_param_knowledge_record.statement_query

        counter_param_knowledge_record.knowledge_probing_prompt = (
            f"{knowledge_probing_prompt_context} {knowledge_probing_prompt_query}"
        )

        # get the last token position (index) for the context object, query subject, and query relation
        object_context_idx, subject_query_idx, relation_query_idx = get_tokens_positions(
            tokenizer=tokenizer,
            input_context=knowledge_probing_prompt_context,
            input_query=knowledge_probing_prompt_query,
            subject_query=counter_param_knowledge_record.statement_subject,
        )

        if _debug_first:
            _debug_token_positions(
                tokenizer=tokenizer,
                input_context=knowledge_probing_prompt_context,
                input_query=knowledge_probing_prompt_query,
                subject_query=counter_param_knowledge_record.statement_subject,
            )
            _debug_first = False

        # register the hooks at the right module
        forward_pass_with_hooks(
            model=model,
            model_name=model_name,
            tokenizer=tokenizer,
            knowledge_probing_prompt=counter_param_knowledge_record.knowledge_probing_prompt,
            device=device,
            first_token_idx=0,
            object_context_idx=object_context_idx,
            subject_query_idx=subject_query_idx,
            relation_query_idx=relation_query_idx,
            include_first_token=include_first_token,
            include_object_context_token=include_object_context_token,
            include_subject_query_token=include_subject_query_token,
            include_relation_query_token=include_relation_query_token,
            include_mlps=include_mlps,
            include_mlps_l1=include_mlps_l1,
            include_mhsa=include_mhsa,
            vertical=vertical,
            activations=activations,
        )
        # Single generate call: output text + all-layer hidden states at the first generated
        # token position. hidden_states[0] is the first step: tuple of (num_layers+1) tensors,
        # each (1, seq_len, hidden_dim). Skip index 0 (embedding); take last token → (num_layers, hidden_dim).
        _inputs = tokenizer(
            counter_param_knowledge_record.knowledge_probing_prompt, return_tensors="pt"
        ).to(device)
        with torch.no_grad():
            _gen_out = model.generate(
                _inputs.input_ids,
                attention_mask=_inputs.attention_mask,
                max_new_tokens=max_gen_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
                return_dict_in_generate=True,
                output_scores=True,
                output_hidden_states=True,
                pad_token_id=tokenizer.eos_token_id,
            )
        output_txt = tokenizer.decode(
            _gen_out.sequences[0][_inputs.input_ids[0].shape[0]:], skip_special_tokens=True
        )
        gen_hidden = torch.stack(
            [hs[0, -1, :].cpu().float() for hs in _gen_out.hidden_states[0]]
        )  # (num_layers+1, hidden_dim) — embedding at [0] then transformer layers
        generation_hidden_states_list.append(gen_hidden)

        output_object = re.split(split_regexp, output_txt)[0]

        knowledge_source = identify_knowledge_source(
            parametric_knowledge_object=counter_param_knowledge_record["parametric_knowledge_object"],
            counter_knowledge_object=counter_param_knowledge_record["counter_knowledge_object"],
            output_object=output_object,
        )
        prompting_results.append(
            {
                "knowledge_source": knowledge_source,
                "output_txt": output_txt,
                "output_object": output_object.lower().strip(),
                "counter_knowledge_object": counter_param_knowledge_record["counter_knowledge_object"].lower().strip(),
                "parametric_knowledge_object": counter_param_knowledge_record["parametric_knowledge_object"]
                .lower()
                .strip(),
                "statement_query": counter_param_knowledge_record["statement_query"],
                "statement_subject": counter_param_knowledge_record["statement_subject"],
                "rel_lemma": counter_param_knowledge_record.rel_lemma,
                "relation_group_id": counter_param_knowledge_record.relation_group_id,
                "prompt_context": knowledge_probing_prompt_context,
                "prompt_query": knowledge_probing_prompt_query,
            }
        )

    prompting_results_df = pd.DataFrame(prompting_results)
    prompting_results_df["generation_hidden_states"] = generation_hidden_states_list

    _print_examples(prompting_results_df, n=10)

    if include_mlps:
        prompting_results_df["mlp"] = activations["mlp"]
    if include_mlps_l1:
        prompting_results_df["mlp-l1"] = activations["mlp-l1"]
    if include_mhsa:
        prompting_results_df["mhsa"] = activations["mhsa"]

    return prompting_results_df
