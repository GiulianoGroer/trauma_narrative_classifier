import numpy as np
import pandas as pd
import numpy as np
import os
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import sent_tokenize, word_tokenize
from huggingface_hub import hf_hub_download
import fasttext
import torch
from transformers import BertTokenizerFast, BertModel
import string

# load nlp models
smodel = SentenceTransformer("all-MiniLM-L6-v2")
wmodel = fasttext.load_model(hf_hub_download("facebook/fasttext-en-vectors", "model.bin"))
tokenizer = BertTokenizerFast.from_pretrained("dbmdz/bert-base-german-cased")
model = BertModel.from_pretrained("dbmdz/bert-base-german-cased")



#### ---- BERT word embeddings ---- ####
def contains_chars(s):
    return (not all(char in string.punctuation or char.isdigit() for char in s))*1

def detect_overlapping_tokens(positions):
    n = len(positions)
    result = [0] * n
    word_index = 1
    previous_token_overlapped = 0
    precede_overlap, next_overlap = 0, 0
    for i in range(n):
        start_i, end_i = positions[i]

        # Check overlap with previous token's end
        if i > 0:
            _, end_prev = positions[i-1]
            if start_i <= end_prev:
                result[i] = word_index
                precede_overlap = 1
            else:
                precede_overlap = 0
        else:
            precede_overlap = 0

        # Check overlap with next token's start
        if i < n - 1:
            start_next, _ = positions[i+1]
            if end_i >= start_next:
                result[i] = word_index
                next_overlap = 1
            else:
                next_overlap = 0
        else:
            next_overlap = 0

        # Increase word index when seeing non-overlapping tokens
        if (not precede_overlap) & (not next_overlap) & previous_token_overlapped:
            word_index += 1

        # store info whether previous token overlapped
        if precede_overlap | next_overlap:
            previous_token_overlapped = 1
        else:
            previous_token_overlapped = 0

    return result

def get_bert_word_embeddings(text, tokenizer, model): #fixed a bug here that would automatically assign the first word the value 0
    """
    Generates word embedding from BERT model.
    """
    encoding = tokenizer.batch_encode_plus(
        [text],
        padding=True,
        truncation=True,
        return_tensors="pt",
        add_special_tokens=False,
        return_offsets_mapping=True
    )

    input_ids = encoding["input_ids"]
    attention_mask = encoding["attention_mask"]

    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        token_embeddings = outputs.last_hidden_state

    tokens = np.array(tokenizer.convert_ids_to_tokens(input_ids[0]))
    overlap_mask = np.array(detect_overlapping_tokens(encoding["offset_mapping"][0]))
    contains_chars_mask = np.array([contains_chars(token) for token in tokens])

    new_words = []
    new_embeddings = []
    seen_overlap_groups = set()

    for token_ind, token in enumerate(tokens):
        overlap = overlap_mask[token_ind]
        is_chars = contains_chars_mask[token_ind]

        if (overlap == 0) & (is_chars):
            # standalone token, not part of a multi-token word group -> always unique, always keep
            emb = token_embeddings[0][token_ind].detach().cpu().numpy()
            new_words.append(token)
            new_embeddings.append(emb)
        elif (overlap != 0) & (is_chars):
            if overlap not in seen_overlap_groups:
                seen_overlap_groups.add(overlap)
                subword_indexer = (overlap_mask == overlap) & (contains_chars_mask == 1)
                this_word = "".join(tokens[subword_indexer])
                this_embedding = token_embeddings[0][subword_indexer].mean(dim=0, keepdim=False)
                this_embedding = this_embedding.detach().cpu().numpy()
                new_words.append(this_word)
                new_embeddings.append(this_embedding)

    new_embeddings = np.stack(new_embeddings, axis=0)
    return new_words, new_embeddings

#### ---- consecutive semantic similarity ---- ####
def consec_sim_features(sim_mat: np.ndarray, k_range: tuple[int]):
    """
    Computes consecutive similarity values (i.e., "coherence") from similarity matrix.

    Parameters
    ----------
    sim_mat : Two-dimensional numpy array
        Cosine similarity matrix of embeddings
    k_range : tuple of ints
        Range of k to compute consecutive similarities for, where k is the inter-word distance

    Returns
    -------
    consec_sim_dict: Dictionary
        dict containing all consecutive similarity descriptives: (Mean, Median, Std, Min, Max)
    """

    # init output dict
    consec_sim_dict = {}

    if sim_mat.ndim != 2:
        raise ValueError("sim_mat must be of dimension 2.")

    if (k_range[0] <= 0) or (k_range[1] <= 0):
        raise ValueError("k_range must contain positive integers.")

    for k in range(*k_range):

        if np.shape(sim_mat)[0] >= k+1: # there should be k+1 words, to give at least one cosine similarity value

            kth_diag = np.diag(sim_mat, k=k) # get the k-th diagonal

            # mean, median, std, min, max
            coh_k_mean = np.mean(kth_diag)
            coh_k_median = np.median(kth_diag)
            coh_k_std = np.std(kth_diag)
            coh_k_min = np.min(kth_diag)
            coh_k_max = np.max(kth_diag)

        else:
            coh_k_mean = coh_k_median = coh_k_std = coh_k_min = coh_k_max = np.nan

        # store data in dict
        consec_sim_dict[str(k)+"_mean"] = coh_k_mean
        consec_sim_dict[str(k)+"_median"] = coh_k_median
        consec_sim_dict[str(k)+"_std"] = coh_k_std
        consec_sim_dict[str(k)+"_min"] = coh_k_min
        consec_sim_dict[str(k)+"_max"] = coh_k_max

    return consec_sim_dict

#### ---- semantic organizaton feature extraction ---- ####
def semantic_organization(text: str):
    """
    Computes various metrics of semantic organization from a given text.

    Parameters
    ----------
    sim_mat : Two-dimensional numpy array
        Cosine similarity matrix of embeddings

    Returns
    -------
    consec_sim_dict: Dictionary
        dict containing all consecutive similarity descriptives: (Mean, Median, Std, Min, Max)
    """

    words = word_tokenize(text)
    sents = sent_tokenize(text)

    sent_vecs = smodel.encode(sents)
    word_vecs = np.stack([wmodel[word] for word in words])
    cwords, cword_vecs = get_bert_word_embeddings(text)

    s_sent_sim = cosine_similarity(sent_vecs)
    w_sent_sim = cosine_similarity(word_vecs)
    cw_sent_sim = cosine_similarity(cword_vecs)

    s_consec_feat_sent = consec_sim_features(sim_mat=s_sent_sim, k_range=(1,3))
    w_consec_feat_sent = consec_sim_features(sim_mat=w_sent_sim, k_range=(1,11))
    cw_consec_feat_sent = consec_sim_features(sim_mat=cw_sent_sim, k_range=(1,11))

    # rename the dict keys appropriately
    s_consec_feat_sent = {"s_consec_"+str(k): v for k, v in s_consec_feat_sent.items()}
    w_consec_feat_sent = {"w_consec_"+str(k): v for k, v in w_consec_feat_sent.items()}
    cw_consec_feat_sent = {"cw_consec_"+str(k): v for k, v in cw_consec_feat_sent.items()}

    # global coherence
    if len(sents) > 1:
        s_low_tril = s_sent_sim[np.tril_indices(s_sent_sim.shape[0], k=-1)]
        s_glob_coherence = {"s_glob_mean": np.mean(s_low_tril), "s_glob_median": np.median(s_low_tril), "s_glob_std": np.std(s_low_tril), "s_glob_min": np.min(s_low_tril), "s_glob_max": np.max(s_low_tril)}
    else:
        s_glob_coherence = {"s_glob_mean": np.nan, "s_glob_median": np.nan, "s_glob_std": np.nan, "s_glob_min": np.nan, "s_glob_max": np.nan}

    if len(words) > 1:
        w_low_tril = w_sent_sim[np.tril_indices(w_sent_sim.shape[0], k=-1)]
        w_glob_coherence = {"w_glob_mean": np.mean(w_low_tril), "w_glob_median": np.median(w_low_tril), "w_glob_std": np.std(w_low_tril), "w_glob_min": np.min(w_low_tril), "w_glob_max": np.max(w_low_tril)}
    else:
        w_glob_coherence = {"w_glob_mean": np.nan, "w_glob_median": np.nan, "w_glob_std": np.nan, "w_glob_min": np.nan, "w_glob_max": np.nan}

    if len(cwords) > 1:
        cw_low_tril = cw_sent_sim[np.tril_indices(cw_sent_sim.shape[0], k=-1)]
        cw_glob_coherence = {"cw_glob_mean": np.mean(cw_low_tril), "cw_glob_median": np.median(cw_low_tril), "cw_glob_std": np.std(cw_low_tril), "cw_glob_min": np.min(cw_low_tril), "cw_glob_max": np.max(cw_low_tril)}
    else:
        cw_glob_coherence = {"cw_glob_mean": np.nan, "cw_glob_median": np.nan, "cw_glob_std": np.nan, "cw_glob_min": np.nan, "cw_glob_max": np.nan}

    verbosity = {"n_words": len(words), "n_sents": len(sents), "l_sents": len(words)/len(sents)}

    return pd.Series(s_consec_feat_sent | w_consec_feat_sent | cw_consec_feat_sent | s_glob_coherence | w_glob_coherence | cw_glob_coherence | verbosity)


