# Detecting concurrent PTSD symptoms from written trauma narratives

Code and data for reproducing the results of "PTSD symptom detection from trauma narratives using natural language processing" (Groer, Plank, Zlomuzica).

## What is in this repository

- `notebooks/classifier_clean_fixed.ipynb`: reproduction notebook for all reported analyses
- `src/classifier_function.py`: nested cross-validation, permutation tests, participant-level bootstrap
- `src/models.py`: loads the embedding models (only needed to regenerate embeddings, see below)
- `data_for_public/dataset_minimal_release.csv`: participant-level data without narrative text and without identifiers
- `data_for_public/oof_scores_embeddings_vs_ngram.csv`: out-of-fold prediction scores of the embedding and n-gram classifiers per participant
- `environment.yml`: software environment

## Data privacy

The written narratives are not shared because they describe traumatic life events. The released file contains the gbert-large embeddings of the narratives (1024 dimensions per narrative condition), questionnaire scores, group labels and derived first-person pronoun measures. It contains no narrative text and no response identifiers.

## Setup

```
conda env create -f environment.yml
conda activate ptsd_narratives
jupyter notebook
```

Open `notebooks/classifier_clean_fixed.ipynb` and run all cells. The notebook locates the repository root itself.

## What runs without the narratives

| Analysis | Runs on the public data |
|---|---|
| Nested cross-validation, trauma / negative / neutral embeddings | yes |
| Subjective-measures classifier | yes |
| DASS-21 subscale and DSM-5 cluster classifiers | yes |
| PCL-5 cutoff sensitivity | yes |
| PCL-5 and DASS-21 correlations | yes |
| First-person pronoun test | yes (from the released derived column) |
| Primary versus inner-selected estimates (Supplementary Table 3) | yes |
| Embeddings versus n-gram comparison | yes (from `oof_scores_embeddings_vs_ngram.csv`) |
| Running the TF-IDF n-gram classifier itself | no, needs the narratives |
| Parsing first-person pronouns with spaCy | no, needs the narratives |
| Narrative length statistics | no, needs the narratives |
| Regenerating the embeddings (`src/models.py`) | no, needs the narratives |

## Permutation tests

Permutation tests (500 permutations, repeating the whole nested cross-validation) are switched off by default (`RUN_PERMUTATIONS = False`) because they take many hours. Set it to `True` to rerun them. The p-values reported in the paper come from these runs.

## Values to check after running

- Trauma classifier (SVM), pooled AUC .734, 95% CI [.646, .819]
- Same analysis without the family choice (inner-selected path), pooled AUC .648 [.542, .748]
- Embeddings versus n-gram, difference in pooled AUC .152, 95% CI [.041, .267]
- First-person subject rate, high versus low PCL-5 group: t = .95, p = .344

## Software

Python 3.11.13, numpy 2.1.3, pandas 2.3.1, scipy 1.16.3, scikit-learn 1.7.2. Analyses on the narratives additionally used spaCy 3.8.14 with de_core_news_lg 3.8.0.
