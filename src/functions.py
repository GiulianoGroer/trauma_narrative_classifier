#function that analyses the text responses for further use

import pandas as pd
import fasttext.util
import nltk
import sys
import os
import re
import numpy as np
import networkx as nx
from textblob_de import TextBlobDE
from sklearn.metrics.pairwise import (cosine_similarity,cosine_distances)
from src.laurins_functions import get_bert_word_embeddings
from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_val_score,
    permutation_test_score,
)
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    roc_auc_score,
    confusion_matrix,
)
from sklearn.pipeline import Pipeline  # Changed from imblearn.pipeline
from scipy.stats import bootstrap
import itertools

def analyze_text_responses(dataset, textvariables, model, model_word, tokenizer_bert, model_bert):
    
    for column_name in textvariables:
        for i in range(len(dataset)):

            text = str(dataset.at[i, column_name])
            blob = TextBlobDE(text)

            # sentiment
            polarity = blob.sentiment.polarity
            subjectivity = blob.sentiment.subjectivity

            # tokens
            sentence_token = [str(s) for s in blob.sentences]
            sentence_count = len(sentence_token)
            
            contextualized_word_embed = get_bert_word_embeddings(text, tokenizer_bert, model_bert)
            

            # ========== CONTEXTUALIZED WORD COHERENCE (BERT) ==========
            if len(contextualized_word_embed[0]) >= 2:
                

                w_cosim_context = cosine_similarity(contextualized_word_embed[1])

                for k in range(1, 11):
                    if len(contextualized_word_embed[0]) >= k + 1:
                        diag = np.diag(w_cosim_context, k=k)
                        dataset.loc[i, f"{column_name}_{k}_order_contextualized_word_coherence_mean"] = np.mean(diag)
                        dataset.loc[i, f"{column_name}_{k}_order_contextualized_word_coherence_median"] = np.median(diag)
                        dataset.loc[i, f"{column_name}_{k}_order_contextualized_word_coherence_std"] = np.std(diag)
                        dataset.loc[i, f"{column_name}_{k}_order_contextualized_word_coherence_min"] = np.min(diag)
                        dataset.loc[i, f"{column_name}_{k}_order_contextualized_word_coherence_max"] = np.max(diag)
                    else:
                        for metric in ["mean","median","std","min","max"]:
                            dataset.loc[i, f"{column_name}_{k}_order_contextualized_word_coherence_{metric}"] = np.nan

            

            # ========== STATIC WORD COHERENCE (FastText) ==========
            if len(contextualized_word_embed[0]) >= 2:
                # static FastText word embeddings
                static_word_embed = np.array([model_word[w] for w in contextualized_word_embed[0]])
                w_cosim_static = cosine_similarity(static_word_embed)

                for k in range(1, 11):
                    if len(contextualized_word_embed[0]) >= k + 1:
                        diag = np.diag(w_cosim_static, k=k)
                        dataset.loc[i, f"{column_name}_{k}_order_static_word_coherence_mean"] = np.mean(diag)
                        dataset.loc[i, f"{column_name}_{k}_order_static_word_coherence_median"] = np.median(diag)
                        dataset.loc[i, f"{column_name}_{k}_order_static_word_coherence_std"] = np.std(diag)
                        dataset.loc[i, f"{column_name}_{k}_order_static_word_coherence_min"] = np.min(diag)
                        dataset.loc[i, f"{column_name}_{k}_order_static_word_coherence_max"] = np.max(diag)
                    else:
                        for metric in ["mean","median","std","min","max"]:
                            dataset.loc[i, f"{column_name}_{k}_order_static_word_coherence_{metric}"] = np.nan


            # ========== SENTENCE COHERENCE ==========
            if sentence_count >= 2:
                sentence_embed = model.encode(sentence_token)
                s_cosim = cosine_similarity(sentence_embed)

                for k in range(1, 11):
                    if sentence_count >= k + 1:
                        diag = np.diag(s_cosim, k=k)
                        dataset.loc[i, f"{column_name}_{k}_order_sentence_coherence_mean"] = np.mean(diag)
                        dataset.loc[i, f"{column_name}_{k}_order_sentence_coherence_median"] = np.median(diag)
                        dataset.loc[i, f"{column_name}_{k}_order_sentence_coherence_std"] = np.std(diag)
                        dataset.loc[i, f"{column_name}_{k}_order_sentence_coherence_min"] = np.min(diag)
                        dataset.loc[i, f"{column_name}_{k}_order_sentence_coherence_max"] = np.max(diag)
                    else:
                        for metric in ["mean", "median", "std", "min", "max"]:
                            dataset.loc[i, f"{column_name}_{k}_order_sentence_coherence_{metric}"] = np.nan

            word_count = len(contextualized_word_embed[0])


            # ========== Simple features ==========
            dataset.loc[i, f"{column_name}_wc"] = word_count
            dataset.loc[i, f"{column_name}_sc"] = sentence_count
            dataset.loc[i, f"{column_name}_polarity"] = polarity
            dataset.loc[i, f"{column_name}_subjectivity"] = subjectivity

    return dataset

def score_questionnaire(dataset, questionnaire, n_items, total_name=None):
    columns = [f"{questionnaire}_{i}" for i in range (1, n_items + 1)]
    for col in columns:
        dataset[col] = pd.to_numeric(dataset[col], errors="coerce")

    if total_name is None:
        total_name = f"{questionnaire}_total"

    dataset[total_name] = dataset[columns].sum(axis=1)
    return dataset

def generate_full_text_embed(dataset, textvariables, model):

    for column_name in textvariables:
        for i in range(len(dataset)):

            text = str(dataset.at[i, column_name]).strip()

            # ========== EMBEDDING OF THE ENTIRE TEXT ==========
            if len(text) > 0:
                full_embedding = model.encode(text)  # vector for whole answer
                for dim_idx, value in enumerate(full_embedding):
                    dataset.loc[i, f"{column_name}_embedding_{dim_idx}"] = value
            else:
                # Empty text -> fill NaN
                dim = model.get_sentence_embedding_dimension()
                for dim_idx in range(dim):
                    dataset.loc[i, f"{column_name}_embedding_{dim_idx}"] = np.nan

    return dataset

def compute_total_semvar_over_responses(dataset, textvariables, model):
    dim = model.get_sentence_embedding_dimension()

    for i in range(len(dataset)):
        row_embs = []

        for column_name in textvariables:
            emb_cols = [f"{column_name}_embedding_{d}" for d in range(dim)]
            emb_values = dataset.loc[i, emb_cols].values

            # skip if this answer is completely missing
            if len(emb_values) == 0 or np.all(pd.isna(emb_values)):
                continue

            row_embs.append(emb_values)

        # if fewer than 2 valid responses: no meaningful variance
        if len(row_embs) < 2:
            dataset.at[i, "total_semvar_over_responses"] = np.nan
            continue

        row_embs = np.vstack(row_embs)        # shape: (n_responses, dim)
        var_per_dim = np.var(row_embs, axis=0)
        mean_var = np.mean(var_per_dim)

        dataset.at[i, "total_semvar_over_responses"] = mean_var

    return dataset



def compute_pairwise_semantic_distances(dataset, textvariables, model):
    dim = model.get_sentence_embedding_dimension()

    for i in range(len(dataset)):
        row_embs = []
        valid_names = []

        for column_name in textvariables:
            emb_cols = [f"{column_name}_embedding_{d}" for d in range(dim)]
            emb_values = dataset.loc[i, emb_cols].values

            # skip if this answer is completely missing
            if len(emb_values) == 0 or np.all(pd.isna(emb_values)):
                continue

            row_embs.append(emb_values)
            valid_names.append(column_name)

        # initialize all pairwise columns with NaN
        for a, b in itertools.combinations(textvariables, 2):
            colname = f"dist_{a}_{b}"
            dataset.at[i, colname] = np.nan

        # need at least 2 responses
        if len(row_embs) < 2:
            continue

        # stack embeddings
        row_embs = np.vstack(row_embs)

        # cosine distance matrix
        dist_matrix = cosine_distances(row_embs)

        # map names to indices
        name_to_idx = {name: idx for idx, name in enumerate(valid_names)}

        # fill distances dynamically
        for a, b in itertools.combinations(textvariables, 2):
            if a in name_to_idx and b in name_to_idx:
                dataset.at[i, f"dist_{a}_{b}"] = \
                    dist_matrix[name_to_idx[a], name_to_idx[b]]

    return dataset

def bootstrap_ci(data, n_resamples=2000, confidence_level=0.95, random_state=0):
    data = np.asarray(data)
    rng = np.random.default_rng(random_state)
    res = bootstrap(
        (data,),
        np.mean,
        vectorized=False,
        n_resamples=n_resamples,
        confidence_level=confidence_level,
        random_state=rng,
    )
    return res.confidence_interval.low, res.confidence_interval.high

def make_pipelines():
    base_lr = lambda C, cw: LogisticRegression(
        solver="liblinear",
        C=C,
        class_weight=cw,
        random_state=0,
        penalty="l2",
        max_iter=1000  # Ensure convergence
    )

    C_grid = [0.01, 0.1, 1.0, 10.0, 100.0]  # Expanded grid for better tuning
    configs = []

    # 1) No imbalance handling (unweighted)
    for C in C_grid:
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("model", base_lr(C=C, cw=None))
        ])
        configs.append(("no_weight", C, pipe))

    # 2) Class-weight balanced (our recommended approach)
    for C in C_grid:
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("model", base_lr(C=C, cw="balanced"))
        ])
        configs.append(("class_weight", C, pipe))

    return configs


def analyze_anchored_vector_similarity(dataset, textvariables, model, anchored_vectors: dict):
    for column_name in textvariables:
        for i in range(len(dataset)):
            text = str(dataset.at[i, column_name])
            if len(text.strip()) >= 1:
                text_embedding = model.encode(text, normalize_embeddings=True)
                for vector_name, anchored_vector in anchored_vectors.items():
                    sim = cosine_similarity(text_embedding.reshape(1, -1), anchored_vector.reshape(1, -1))[0][0]
                    dataset.loc[i, f"{column_name}_anchored_vector_{vector_name}_cosine_sim"] = sim
            else:
                for vector_name in anchored_vectors.keys():
                    dataset.loc[i, f"{column_name}_anchored_vector_{vector_name}_cosine_sim"] = np.nan
    return dataset



def create_anchored_vector(negative_phrases: list[str], positive_phrases: list[str], model):
    neg = model.encode(negative_phrases, normalize_embeddings=True).mean(axis=0)
    pos = model.encode(positive_phrases, normalize_embeddings=True).mean(axis=0)
    anchored_vector = pos - neg
    anchored_vector = anchored_vector / np.linalg.norm(anchored_vector)
    return anchored_vector


def get_questionnaire_item_embeddings(items: list[str], model):
    embeddings = model.encode(items)
    return {idx: embedding for idx, embedding in enumerate(embeddings)}

def analyze_questionnaire_item_similarity(dataset, textvariables, model, item_embeddings: dict):
    """
    Parameters
    ----------
    dataset : pd.DataFrame
    textvariables : list of str
    model : SentenceTransformer
        SentenceTransformer model used to encode sentences
    item_embeddings : dict
        Dictionary of named questionnaire item embeddings e.g.
        {0: embedding_array, 1: embedding_array, ...}
        as returned by get_questionnaire_item_embeddings()
    """

    for column_name in textvariables:
        for i in range(len(dataset)):
            text = str(dataset.at[i, column_name])

            if len(text) >= 1:
                text_embedding = model.encode(text)

                for item_idx, item_embedding in item_embeddings.items():
                    sim = cosine_similarity(text_embedding.reshape(1, -1), item_embedding.reshape(1, -1))[0][0]
                    dataset.loc[i, f"{column_name}_item_{item_idx}_cosine_sim"] = sim
            else:
                for item_idx in item_embeddings.keys():
                    dataset.loc[i, f"{column_name}_item_{item_idx}_cosine_sim"] = np.nan

    return dataset