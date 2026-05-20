import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

from joblib import Parallel, delayed

from src.classification.data import get_features_and_targets_for_dataset, undersample_dataset


def _fit_regression(train_activations, train_targets, test_activations, test_targets, layer, relation_group_id):
    if test_activations.shape[0] > 0:
        predictive_model = LogisticRegression(random_state=16, max_iter=10000, n_jobs=1)
        predictive_model.fit(train_activations[:, layer, 0, :], train_targets)

        predictions = predictive_model.predict(test_activations[:, layer, 0, :])
        labels = sorted(set(test_targets) | set(predictions))
        cm = confusion_matrix(test_targets, predictions, labels=labels)
        # labels are sorted strings e.g. ["CK", "PK"]; rows=true, cols=pred
        # labels sorted as ["CK", "PK"]: ravel() = [tp, fn, fp, tn] (CK = positive class)
        tp, fn, fp, tn = (cm.ravel() if cm.shape == (2, 2) else (None, None, None, None))
        return {
            "layer": layer,
            "relation_group_id": relation_group_id,
            "accuracy": accuracy_score(test_targets, predictions),
            "macro_f1": f1_score(test_targets, predictions, average="macro", zero_division=0),
            "nb_test_examples": test_activations.shape[0],
            "nb_train_examples": train_activations.shape[0],
            "tn": int(tn) if tn is not None else None,
            "fp": int(fp) if fp is not None else None,
            "fn": int(fn) if fn is not None else None,
            "tp": int(tp) if tp is not None else None,
        }

    else:
        return {
            "layer": None,
            "relation_group_id": relation_group_id,
            "accuracy": None,
            "macro_f1": None,
            "nb_test_examples": None,
            "nb_train_examples": None,
            "tn": None,
            "fp": None,
            "fn": None,
            "tp": None,
        }


def perform_classification_by_relation_group(
    prompting_results: pd.DataFrame,
    include_mlps: bool,
    include_mlps_l1: bool,
    include_mhsa: bool,
    vertical: bool,
    position: str,
) -> None:
    """this function performs classification over the saved activations, the classification can be module-wise and token-wise (i.e. vertical)

    Args:
        prompting_results (pd.DataFrame): the dataframe with all the activations resulting from the prompting
        include_mlps (bool): whether to include the MLPs or not
        include_mlps_l1 (bool): whether to include the first layer of the MLPs or not
        include_mhsa (bool): whether to include the MHSAs or not
        vertical (bool): whether to train the classifier vertically (by token) or all at once
        position (str): the token where the classification is performed (subject_query, object, or relation_query)
    """

    label_encoder = {"CK": 0, "PK": 1}

    prompting_results = prompting_results.reset_index()

    classification_metrics = []

    for relation_group_id in prompting_results.relation_group_id.unique():
        print(f"Evaluating the {relation_group_id} relation group.")

        train_dataset = undersample_dataset(prompting_results[prompting_results.relation_group_id != relation_group_id])
        test_dataset = undersample_dataset(prompting_results[prompting_results.relation_group_id == relation_group_id])

        train_activations, train_targets = get_features_and_targets_for_dataset(
            dataset=train_dataset,
            vertical=vertical,
            position=position,
            include_mlps=include_mlps,
            include_mlps_l1=include_mlps_l1,
            include_mhsa=include_mhsa,
            label_encoder=label_encoder,
        )

        test_activations, test_targets = get_features_and_targets_for_dataset(
            dataset=test_dataset,
            vertical=vertical,
            position=position,
            include_mlps=include_mlps,
            include_mlps_l1=include_mlps_l1,
            include_mhsa=include_mhsa,
            label_encoder=label_encoder,
        )

        classification_metrics += Parallel(n_jobs=12, backend="loky", verbose=10)(
            delayed(_fit_regression)(
                train_activations=train_activations,
                train_targets=train_targets,
                test_activations=test_activations,
                test_targets=test_targets,
                layer=layer,
                relation_group_id=relation_group_id,
            )
            for layer in list(range(train_activations.shape[1]))
        )

    return classification_metrics


def print_classification_summary(classification_metrics: list) -> None:
    df = pd.DataFrame(classification_metrics).dropna(subset=["accuracy"])

    # best layer: highest accuracy averaged across relation groups, weighted by group size
    by_layer_acc = df.groupby("layer").apply(
        lambda g: np.average(g["accuracy"], weights=g["nb_test_examples"])
    )
    by_layer_f1 = df.groupby("layer").apply(
        lambda g: np.average(g["macro_f1"], weights=g["nb_test_examples"])
    )
    best_layer = int(by_layer_acc.idxmax())
    best_layer_acc = by_layer_acc.max()
    best_layer_f1 = by_layer_f1[best_layer]
    mean_acc = by_layer_acc.mean()
    mean_f1 = by_layer_f1.mean()

    print("\n" + "=" * 50)
    print("CLASSIFICATION SUMMARY")
    print("=" * 50)
    print(f"Weighted mean accuracy across all layers:  {mean_acc:.3f}")
    print(f"Weighted mean macro-F1 across all layers:  {mean_f1:.3f}")
    print(f"Best layer: {best_layer:>3d}  |  accuracy: {best_layer_acc:.3f}  |  macro-F1: {best_layer_f1:.3f}")

    print("\nPer relation group (best layer accuracy):")
    per_group = df.groupby("relation_group_id")["accuracy"].max()
    for group, acc in per_group.sort_values(ascending=False).items():
        print(f"  {group:<40s}  {acc:.3f}")
    print("=" * 50 + "\n")


def save_classification_metrics(
    classification_metrics: pd.DataFrame,
    model_name: str,
    permanent_path: str,
    include_mlps: bool,
    include_mlps_l1: bool,
    include_mhsa: bool,
    vertical: bool,
    position: str,
    test_on_head: bool,
    n_head: int,
) -> None:
    """this function saves the classification results into a csv file.

    Args:
        classification_metrics (pd.DataFrame): the classification metrics to save
        model_name (str): the name of the current LLM
        permanent_path (str): the path where to store the results
        include_mlps (bool): whether to include the MLPs or not
        include_mlps_l1 (bool): whether to include the first layer of the MLPs or not
        include_mhsa (bool): whether to include the MHSAs or not
        vertical (bool): whether to train the classifier vertically (by token) or all at once
        position (str): the token where the classification is performed (subject_query, object, or relation_query)
        test_on_head (bool): whether the current experiment is applied on the header of the reformatted ParaRel dataset
        n_head (int): the number of rows to consider on the header (if test_on_head is True)
    """
    # save the classification metrics
    save_path = f"{permanent_path}/classification_metrics/{model_name}_classification_metrics_by_relation_groups"

    if vertical:
        save_path += "_" + position
    else:
        save_path += "_all_positions"

    if test_on_head:
        save_path += f"_n_head={n_head}"

    # Append the appropriate suffix based on the included modules
    if include_mhsa:
        save_path += "_mhsa"
    if include_mlps:
        save_path += "_mlps"
    if include_mlps_l1:
        save_path += "_mlps_l1"
    save_path += ".csv"

    pd.DataFrame(classification_metrics).rename(columns={"accuracy": "accuracy"}).to_csv(save_path, index=False)
