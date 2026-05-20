import torch
import numpy as np
from typing import List, Union
import matplotlib.pyplot as plt
from imblearn.pipeline import Pipeline
from imblearn.under_sampling import RandomUnderSampler
from transformers import AutoTokenizer, AutoModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from sklearn.decomposition import PCA
from RACDH.config import params

def classify_texts(texts: List[str], labels: List[Union[str,int]], 
                   imbalance_mode: str = "undersample",
                   model_name: str = "microsoft/deberta-v3-large"):
    """
    Classify using both:
      A) Bag-of-Words (TF-IDF) 
      B) Transformer Embeddings
    Handling class imbalance via either random undersampling or class-weight.
    """
    # 1. Build BOW Pipeline
    if imbalance_mode == "undersample":
        bow_pipe = Pipeline([
            ("tfidf", TfidfVectorizer(ngram_range=(1,2), max_features=5000)),
            ("undersampler", RandomUnderSampler(random_state=2)),
            ("clf", LogisticRegression(random_state=42, max_iter=1000))
        ])
    elif imbalance_mode == "balanced":
        bow_pipe = Pipeline([
            ("tfidf", TfidfVectorizer(ngram_range=(1,2), max_features=5000)),
            ("clf", LogisticRegression(class_weight="balanced", random_state=42, max_iter=1000))
        ])
    else:
        bow_pipe = Pipeline([
            ("tfidf", TfidfVectorizer(ngram_range=(1,2), max_features=5000)),
            ("clf", LogisticRegression(random_state=42, max_iter=1000))
        ])
    
    # 2. Run cross-val + train/test on BOW
    print(f"--- BOW Classification [{imbalance_mode}] [last sentence: {params.last_sentence}] ---")
    _run_cv_and_split(bow_pipe, texts, labels)

    # 3. Generate Embeddings
    embeddings = _get_embeddings(texts, model_name=model_name)

    # (Optional) PCA Plot for Embeddings
    # comment out if you don't need a plot
    _plot_pca(embeddings, labels)

    # 4. Build Pipeline for Embeddings
    if imbalance_mode == "undersample":
        emb_pipe = Pipeline([
            ("undersampler", RandomUnderSampler(random_state=2)),
            ("clf", LogisticRegression(random_state=42, max_iter=1000))
        ])
    else:
        emb_pipe = Pipeline([
            ("clf", LogisticRegression(random_state=42, max_iter=1000))
        ])

    # 5. Run cross-val + train/test on Embeddings
    print(f"\n--- Embeddings Classification [{imbalance_mode}] [last sentence: {params.last_sentence}] ---")
    _run_cv_and_split(emb_pipe, embeddings, labels)


def _run_cv_and_split(pipe, X, y):
    X_arr = np.array(X) if not isinstance(X, np.ndarray) else X
    y_arr = np.array(y)
    classes = sorted(set(y))

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=2)
    per_class = {c: {"precision": [], "recall": [], "f1": []} for c in classes}
    macro = {"precision": [], "recall": [], "f1": []}
    accuracies = []

    for train_idx, test_idx in skf.split(X_arr, y_arr):
        p_clone = clone(pipe)
        p_clone.fit(X_arr[train_idx], y_arr[train_idx])
        y_pred = p_clone.predict(X_arr[test_idx])
        y_te = y_arr[test_idx]

        accuracies.append(accuracy_score(y_te, y_pred))
        prec, rec, f1, _ = precision_recall_fscore_support(
            y_te, y_pred, labels=classes, zero_division=0
        )
        for i, c in enumerate(classes):
            per_class[c]["precision"].append(prec[i])
            per_class[c]["recall"].append(rec[i])
            per_class[c]["f1"].append(f1[i])
        pm, rm, fm, _ = precision_recall_fscore_support(
            y_te, y_pred, average="macro", zero_division=0
        )
        macro["precision"].append(pm)
        macro["recall"].append(rm)
        macro["f1"].append(fm)

    print("\n[5-Fold Cross-Validation (mean ± std)]")
    print(f"{'':>16}  {'precision':>14}  {'recall':>14}  {'f1-score':>14}")
    for c in classes:
        p = np.array(per_class[c]["precision"])
        r = np.array(per_class[c]["recall"])
        f = np.array(per_class[c]["f1"])
        print(f"{str(c):>16}  {p.mean():.3f} ± {p.std():.3f}  {r.mean():.3f} ± {r.std():.3f}  {f.mean():.3f} ± {f.std():.3f}")
    pm = np.array(macro["precision"])
    rm = np.array(macro["recall"])
    fm = np.array(macro["f1"])
    acc = np.array(accuracies)
    print(f"{'macro avg':>16}  {pm.mean():.3f} ± {pm.std():.3f}  {rm.mean():.3f} ± {rm.std():.3f}  {fm.mean():.3f} ± {fm.std():.3f}")
    print(f"{'accuracy':>16}  {acc.mean():.3f} ± {acc.std():.3f}")

    # Fit on full data so _print_top_bow_features_and_counts can inspect coefficients
    pipe.fit(X_arr, y_arr)

    # Show top BOW features + token frequency stats if this is a TF-IDF pipeline
    _print_top_bow_features_and_counts(pipe, X, y, n=20)


def _print_top_bow_features_and_counts(pipe, X, y, n=20):
    """
    1) If the pipeline includes a TF-IDF vectorizer + LogisticRegression,
       print the top weighted tokens for each class.
    2) For each top token, count how often it appears in parametric vs. contextual texts.
    """
    if "tfidf" not in pipe.named_steps:
        return  # Not a BOW pipeline

    vectorizer = pipe.named_steps["tfidf"]
    clf = pipe.named_steps["clf"]
    feature_names = vectorizer.get_feature_names_out()

    coefs = clf.coef_
    if coefs.shape[0] == 1:
        # Binary classification
        coefs = coefs[0]  # Flatten to 1D
        classes = clf.classes_
        class_0 = classes[0]
        class_1 = classes[1]

        top_pos_indices = np.argsort(coefs)[::-1][:n]
        top_neg_indices = np.argsort(coefs)[:n]

        print(f"\nTop {n} tokens indicating class '{class_1}' (largest +coef):")
        pos_tokens = []
        for idx in top_pos_indices:
            token = feature_names[idx]
            weight = coefs[idx]
            pos_tokens.append(token)
            print(f"   {token}: {weight:.4f}")

        print(f"\nTop {n} tokens indicating class '{class_0}' (most negative coef):")
        neg_tokens = []
        for idx in top_neg_indices:
            token = feature_names[idx]
            weight = coefs[idx]
            neg_tokens.append(token)
            print(f"   {token}: {weight:.4f}")

        # Count how often these tokens appear in each class
        print("\n=== Token Frequency in Parametric vs. Contextual ===")
        _count_token_occurrences(pos_tokens, X, y, class_0, class_1, "parametric")
        _count_token_occurrences(neg_tokens, X, y, class_0, class_1, "contextual")
    else:
        print("Top features display only implemented for binary Logistic Regression.")


def _count_token_occurrences(token_list, texts, labels, class_0, class_1, name):
    """
    For each token in token_list, print how often it appears in class_0 vs class_1 texts.
    Note: This is a simple substring check: if token in text.lower().
          If you want exact match, do actual tokenization.
    """
    # Convert everything to lower for case-insensitive matching
    lower_texts = [t.lower() for t in texts]

    # Indices for each class
    indices_0 = [i for i, lab in enumerate(labels) if lab == class_0]
    indices_1 = [i for i, lab in enumerate(labels) if lab == class_1]

    texts_0 = [lower_texts[i] for i in indices_0]
    texts_1 = [lower_texts[i] for i in indices_1]

    n_0 = len(texts_0)
    n_1 = len(texts_1)

    print(f"\nToken frequency for tokens indicative of class '{name}':")
    for tok in token_list:
        count_0 = sum(tok in doc for doc in texts_0)
        count_1 = sum(tok in doc for doc in texts_1)

        pct_0 = (count_0 / n_0 * 100) if n_0 > 0 else 0
        pct_1 = (count_1 / n_1 * 100) if n_1 > 0 else 0

        print(f"  '{tok}' -> {class_0}: {count_0}/{n_0} ({pct_0:.1f}%) | {class_1}: {count_1}/{n_1} ({pct_1:.1f}%)")


def _get_embeddings(texts: List[str], model_name: str) -> np.ndarray:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name, torch_dtype=torch.float16).to(device)
    model.eval()
    model.requires_grad_(False)

    all_embs = []
    for text in texts:
        inp = tokenizer(text, max_length=128, padding='max_length',
                        truncation=True, return_tensors='pt')
        for k, v in inp.items():
            inp[k] = v.to(device)
        with torch.no_grad():
            out = model(**inp)
            # Use [CLS] embedding (first token) as the text representation
            cls_vec = out.last_hidden_state[:, 0, :]
            all_embs.append(cls_vec.squeeze(0).cpu().numpy())

    return np.array(all_embs)

def _plot_pca(embeddings: np.ndarray, labels: List[Union[str,int]]):
    pca = PCA(n_components=2)
    emb_2d = pca.fit_transform(embeddings)

    plt.figure(figsize=(6, 5))
    uniq_labels = sorted(set(labels))
    for lbl in uniq_labels:
        idxs = [i for i, y in enumerate(labels) if y == lbl]
        plt.scatter(emb_2d[idxs, 0], emb_2d[idxs, 1], label=lbl)
    plt.title("PCA of Embeddings")
    plt.legend()
    plt.show()
