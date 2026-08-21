import numpy as np
from collections import Counter
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.dummy import DummyClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, accuracy_score
from scipy.stats import bootstrap
from sklearn.feature_extraction.text import TfidfVectorizer

# =====================================================
# Bootstrap CI helper
# =====================================================
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

def participant_level_single_ci(y_true, y_score, n_resamples=2000, random_state=0):
    rng = np.random.default_rng(random_state)
    n = len(y_true)
    aucs = np.empty(n_resamples)
    for i in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        y_r, s_r = y_true[idx], y_score[idx]
        if len(np.unique(y_r)) < 2:
            aucs[i] = np.nan
            continue
        aucs[i] = roc_auc_score(y_r, s_r)
    aucs = aucs[~np.isnan(aucs)]
    return aucs.mean(), np.percentile(aucs, 2.5), np.percentile(aucs, 97.5)

def participant_level_paired_bootstrap(y_true, score_a, score_b, n_resamples=2000, random_state=0):
    rng = np.random.default_rng(random_state)
    n = len(y_true)
    diffs = np.empty(n_resamples)
    for i in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        y_r, a_r, b_r = y_true[idx], score_a[idx], score_b[idx]
        if len(np.unique(y_r)) < 2:
            diffs[i] = np.nan
            continue
        diffs[i] = roc_auc_score(y_r, a_r) - roc_auc_score(y_r, b_r)
    diffs = diffs[~np.isnan(diffs)]
    return diffs.mean(), np.percentile(diffs, 2.5), np.percentile(diffs, 97.5), diffs

# =====================================================
# Candidate pipelines across three model families
# =====================================================
def make_pipelines_ngram():
    pipelines_list = []
    for weighting in ["no_weight", "class_weight"]:
        cw = None if weighting == "no_weight" else "balanced"
        for ngram_range in [(1, 1), (1, 2)]:
            for C in [0.01, 0.1, 1.0, 10.0]:
                pipe = Pipeline([
                    ("vectorizer", TfidfVectorizer(
                        ngram_range=ngram_range,
                        min_df=2,          # drop hapax terms, given N≈150
                        max_features=2000  # cap dimensionality at this sample size
                    )),
                    ("model", LogisticRegression(
                        solver="liblinear", C=C, class_weight=cw,
                        random_state=0, penalty="l2", max_iter=1000
                    ))
                ])
                pipelines_list.append(("ngram", f"{weighting}_{ngram_range}", C, pipe))
    return pipelines_list

def make_pipelines():
    pipelines_list = []

    for weighting in ["no_weight", "class_weight"]:
        cw = None if weighting == "no_weight" else "balanced"
        for C in [0.01, 0.1, 1.0, 10.0, 100.0]:
            pipe = Pipeline([
                ("scaler", StandardScaler()),
                ("model", LogisticRegression(
                    solver="liblinear", C=C, class_weight=cw,
                    random_state=0, penalty="l2", max_iter=1000
                ))
            ])
            pipelines_list.append(("logreg", weighting, C, pipe))

    for weighting in ["no_weight", "class_weight"]:
        cw = None if weighting == "no_weight" else "balanced"
        for C in [0.01, 0.1, 1.0, 10.0]:
            pipe = Pipeline([
                ("scaler", StandardScaler()),
                ("model", LinearSVC(
                    C=C, class_weight=cw, random_state=0, max_iter=5000
                ))
            ])
            pipelines_list.append(("svm", weighting, C, pipe))

    for n_estimators, max_depth in [(100, 2), (100, 3), (200, 2), (200, 3)]:
        pipe = Pipeline([
            ("model", GradientBoostingClassifier(
                n_estimators=n_estimators, max_depth=max_depth,
                random_state=0
            ))
        ])
        pipelines_list.append(("gbt", f"depth{max_depth}", n_estimators, pipe))

    return pipelines_list


def _get_scores(fitted_pipe, X):
    """Returns a rankable score for roc_auc_score regardless of model type."""
    model = fitted_pipe.named_steps["model"]
    if hasattr(model, "predict_proba"):
        return fitted_pipe.predict_proba(X)[:, 1]
    else:
        return fitted_pipe.decision_function(X)


# =====================================================
# Generic nested CV engine
# =====================================================
def run_nested_cv(X_data, y_data, pipelines_list,
                   n_outer_splits=5, n_inner_splits=5,
                   outer_random_state=0, inner_random_state=1,
                   verbose=True):

    outer_cv = StratifiedKFold(n_splits=n_outer_splits, shuffle=True, random_state=outer_random_state)
    inner_cv = StratifiedKFold(n_splits=n_inner_splits, shuffle=True, random_state=inner_random_state)

    outer_auc_scores = []
    outer_acc_scores = []
    outer_chosen = []
    dummy_auc_scores = []
    outer_y_true = []
    outer_y_score = []

    family_names = sorted(set(fam for fam, *_ in pipelines_list))
    per_family_auc = {fam: [] for fam in family_names}
    per_family_chosen = {fam: [] for fam in family_names}
    per_family_test_idx = {fam: [] for fam in family_names}
    per_family_y_true = {fam: [] for fam in family_names}
    per_family_y_score = {fam: [] for fam in family_names}

    # ============================================================
    # OUTER FOLD LOOP — everything in this block runs PER FOLD
    # ============================================================
    for outer_fold, (outer_train_idx, outer_test_idx) in enumerate(outer_cv.split(X_data, y_data)):
        if verbose:
            print(f"\n--- Outer Fold {outer_fold + 1} ---")
        X_tr, X_te = X_data[outer_train_idx], X_data[outer_test_idx]
        y_tr, y_te = y_data[outer_train_idx], y_data[outer_test_idx]

        best_auc = -np.inf
        best_cfg = None
        best_pipe = None

        family_best_auc = {fam: -np.inf for fam in family_names}
        family_best_pipe = {fam: None for fam in family_names}
        family_best_cfg = {fam: None for fam in family_names}

        for family, label, param, pipe in pipelines_list:
            cv_auc = cross_val_score(pipe, X_tr, y_tr, cv=inner_cv, scoring="roc_auc", n_jobs=-1).mean()

            if cv_auc > best_auc:
                best_auc, best_cfg, best_pipe = cv_auc, (family, label, param), pipe

            if cv_auc > family_best_auc[family]:
                family_best_auc[family] = cv_auc
                family_best_pipe[family] = pipe
                family_best_cfg[family] = (label, param)

        best_pipe.fit(X_tr, y_tr)
        y_score = _get_scores(best_pipe, X_te)
        y_pred = best_pipe.predict(X_te)

        auc_outer = roc_auc_score(y_te, y_score)
        acc_outer = accuracy_score(y_te, y_pred)
        outer_auc_scores.append(auc_outer)
        outer_acc_scores.append(acc_outer)
        outer_chosen.append(best_cfg)
        outer_y_true.append(y_te)
        outer_y_score.append(y_score)

        if verbose:
            print(f"Overall best inner: {best_cfg}, inner AUC={best_auc:.3f}")
            print(f"Overall outer test: AUC={auc_outer:.3f}, Acc={acc_outer:.3f}")

        for fam in family_names:
            fam_pipe = family_best_pipe[fam]
            fam_pipe.fit(X_tr, y_tr)
            fam_score = _get_scores(fam_pipe, X_te)
            fam_auc = roc_auc_score(y_te, fam_score)
            per_family_auc[fam].append(fam_auc)
            per_family_chosen[fam].append(family_best_cfg[fam])
            per_family_test_idx[fam].append(outer_test_idx)
            per_family_y_true[fam].append(y_te)
            per_family_y_score[fam].append(fam_score)
            if verbose:
                print(f"  [{fam}] best inner cfg={family_best_cfg[fam]}, outer AUC={fam_auc:.3f}")

        dummy = DummyClassifier(strategy="most_frequent")
        dummy.fit(X_tr, y_tr)
        dummy_auc = roc_auc_score(y_te, dummy.predict_proba(X_te)[:, 1])
        dummy_auc_scores.append(dummy_auc)
    # ============================================================
    # LOOP ENDS HERE — everything below runs ONCE, after all folds
    # ============================================================

    pooled_y_true = np.concatenate(outer_y_true)
    pooled_y_score = np.concatenate(outer_y_score)

    auc_low, auc_high = bootstrap_ci(np.array(outer_auc_scores))

    per_family_pooled = {}
    for fam in family_names:
        idx_concat = np.concatenate(per_family_test_idx[fam])
        y_true_concat = np.concatenate(per_family_y_true[fam])
        y_score_concat = np.concatenate(per_family_y_score[fam])
        order = np.argsort(idx_concat)
        per_family_pooled[fam] = {
            "participant_idx": idx_concat[order],
            "y_true": y_true_concat[order],
            "y_score": y_score_concat[order],
        }

    results = {
        "outer_auc_scores": np.array(outer_auc_scores),
        "outer_acc_scores": np.array(outer_acc_scores),
        "outer_chosen": outer_chosen,
        "dummy_auc_scores": np.array(dummy_auc_scores),
        "auc_ci": (auc_low, auc_high),
        "per_family_auc": {fam: np.array(v) for fam, v in per_family_auc.items()},
        "per_family_chosen": per_family_chosen,
        "per_family_pooled": per_family_pooled,
        "pooled_y_true": pooled_y_true,
        "pooled_y_score": pooled_y_score,
        "outer_cv": outer_cv,
    }

    if verbose:
        print("\n=== SUMMARY ===")
        print(f"Overall Mean Outer AUC: {np.mean(outer_auc_scores):.3f} ± {np.std(outer_auc_scores):.3f}")
        print(f"95% CI: [{auc_low:.3f}, {auc_high:.3f}]")
        for fam in family_names:
            v = results["per_family_auc"][fam]
            print(f"[{fam}] Mean Outer AUC: {v.mean():.3f} ± {v.std():.3f}")

    return results


# =====================================================
# Orchestrator: runs all narrative conditions with matched folds
# =====================================================
def classifier_builder_multi(dataset, target_col, positive_class,
                              response_types=("response_trauma", "response_negnt", "response_neutr"),
                              n_outer_splits=5, n_inner_splits=5,
                              outer_random_state=0, inner_random_state=1):

    embedding_cols_by_condition = {
        rt: [c for c in dataset.columns if c.startswith(rt) and "_embedding_" in c]
        for rt in response_types
    }

    all_embedding_cols = [c for cols in embedding_cols_by_condition.values() for c in cols]
    complete_dataset = dataset.dropna(subset=all_embedding_cols).reset_index(drop=True)

    y_data = ((complete_dataset[target_col] == positive_class) * 1).to_numpy()
    print(f"Complete-case N (valid embeddings across all conditions): {len(y_data)}")
    print("Positive class proportion:", y_data.mean())

    pipelines_list = make_pipelines()

    condition_results = {}

    for rt in response_types:
        print(f"\n{'='*60}\nCONDITION: {rt}\n{'='*60}")
        X_data = complete_dataset[embedding_cols_by_condition[rt]].to_numpy()

        results = run_nested_cv(
            X_data, y_data, pipelines_list,
            n_outer_splits=n_outer_splits, n_inner_splits=n_inner_splits,
            outer_random_state=outer_random_state, inner_random_state=inner_random_state,
        )
        condition_results[rt] = results

    return condition_results, complete_dataset, embedding_cols_by_condition, y_data, pipelines_list


# =====================================================
# Best-family selection and pairwise comparison
# (compares each condition's own best-performing family,
#  not a pooled/mixed-family selection)
# =====================================================
def pick_best_family(results):
    means = {fam: v.mean() for fam, v in results["per_family_auc"].items()}
    best_fam = max(means, key=means.get)
    return best_fam, results["per_family_auc"][best_fam]


def compare_best_families(condition_results, reference="response_trauma"):
    best = {}
    for rt, res in condition_results.items():
        fam, scores = pick_best_family(res)
        best[rt] = (fam, scores)
        print(f"{rt}: best family = {fam}, mean AUC = {scores.mean():.3f} ± {scores.std():.3f}")

    ref_fam, ref_scores = best[reference]
    pairwise = {}
    print(f"\nPairwise comparisons (reference: {reference} [{ref_fam}])")
    for rt, (fam, scores) in best.items():
        if rt == reference:
            continue
        diff = ref_scores - scores
        diff_low, diff_high = bootstrap_ci(diff)
        sig = diff_low > 0 or diff_high < 0
        print(f"\n{reference} ({ref_fam}) vs {rt} ({fam})")
        print(f"  Per-fold diff: {np.round(diff, 3)}")
        print(f"  Mean diff: {diff.mean():.3f}, 95% CI: [{diff_low:.3f}, {diff_high:.3f}]")
        print(f"  {'Significant' if sig else 'Not significant'} (CI {'excludes' if sig else 'includes'} 0)")
        pairwise[f"{reference}_vs_{rt}"] = {
            "reference_family": ref_fam, "comparison_family": fam,
            "diff_scores": diff, "mean_diff": diff.mean(), "ci": (diff_low, diff_high),
        }
    return best, pairwise


# =====================================================
# Properly nested permutation test
# (rerun the entire selection+evaluation procedure per permutation;
#  restrict pipelines_list to one family to keep cost manageable)
# =====================================================
def permutation_test_nested_cv(X_data, y_data, pipelines_list,
                                n_outer_splits=5, n_inner_splits=5,
                                outer_random_state=0, inner_random_state=1,
                                n_permutations=500, random_state=0,
                                verbose=True):

    outer_cv = StratifiedKFold(n_splits=n_outer_splits, shuffle=True, random_state=outer_random_state)
    inner_cv = StratifiedKFold(n_splits=n_inner_splits, shuffle=True, random_state=inner_random_state)

    def _nested_mean_auc(y):
        aucs = []
        for tr_idx, te_idx in outer_cv.split(X_data, y):
            X_tr, X_te = X_data[tr_idx], X_data[te_idx]
            y_tr, y_te = y[tr_idx], y[te_idx]
            best_auc, best_pipe = -np.inf, None
            for family, label, param, pipe in pipelines_list:
                cvs = cross_val_score(pipe, X_tr, y_tr, cv=inner_cv, scoring="roc_auc", n_jobs=-1).mean()
                if cvs > best_auc:
                    best_auc, best_pipe = cvs, pipe
            best_pipe.fit(X_tr, y_tr)
            score = _get_scores(best_pipe, X_te)
            aucs.append(roc_auc_score(y_te, score))
        return np.mean(aucs)

    observed = _nested_mean_auc(y_data)

    rng = np.random.RandomState(random_state)
    perm_scores = np.empty(n_permutations)
    for i in range(n_permutations):
        y_perm = rng.permutation(y_data)
        perm_scores[i] = _nested_mean_auc(y_perm)
        if verbose and (i + 1) % 20 == 0:
            print(f"  permutation {i + 1}/{n_permutations}")

    pvalue = (np.sum(perm_scores >= observed) + 1) / (n_permutations + 1)
    if verbose:
        print(f"\nObserved nested mean outer AUC: {observed:.3f}")
        print(f"Mean permuted AUC: {perm_scores.mean():.3f}")
        print(f"p-value: {pvalue:.4f}  (floor given n_permutations={n_permutations}: {1/(n_permutations+1):.4f})")

    return observed, perm_scores, pvalue

def run_symptom_cluster_classifiers(dataset, response_type, cluster_targets,
                                     n_outer_splits=5, n_inner_splits=5,
                                     outer_random_state=0, inner_random_state=1):
    """
    cluster_targets: dict like {
        "Intrusion": ("intrusion_group", "high"),
        "Avoidance": ("avoidance_group", "high"),
        "NegCognitionsMood": ("negcog_group", "high"),
        "ArousalReactivity": ("arousal_group", "high"),
    }
    """
    embedding_cols = [c for c in dataset.columns if c.startswith(response_type) and "_embedding_" in c]
    pipelines_list = make_pipelines()

    cluster_results = {}
    perm_results = {}

    for cluster, (target_col, positive_class) in cluster_targets.items():
        print(f"\n{'='*60}\nDSM-5 CLUSTER: {cluster}\n{'='*60}")

        sub_dataset = dataset.dropna(subset=embedding_cols + [target_col]).reset_index(drop=True)
        X_data = sub_dataset[embedding_cols].to_numpy()
        y_data_sub = ((sub_dataset[target_col] == positive_class) * 1).to_numpy()

        print(f"N = {len(y_data_sub)}, positive proportion = {y_data_sub.mean():.3f}")

        results = run_nested_cv(
            X_data, y_data_sub, pipelines_list,
            n_outer_splits=n_outer_splits, n_inner_splits=n_inner_splits,
            outer_random_state=outer_random_state, inner_random_state=inner_random_state,
        )
        cluster_results[cluster] = results

        fam, _ = pick_best_family(results)
        fam_pipelines = [p for p in pipelines_list if p[0] == fam]

        print(f"\n--- Permutation test: {cluster} (best family: {fam}) ---")
        obs, perm_scores, pval = permutation_test_nested_cv(
            X_data, y_data_sub, fam_pipelines, n_permutations=500
        )
        perm_results[cluster] = {"observed": obs, "perm_scores": perm_scores, "pvalue": pval, "family": fam}

    return cluster_results, perm_results

def run_dass_subscale_classifiers(dataset, response_type, dass_targets,
                                   n_outer_splits=5, n_inner_splits=5,
                                   outer_random_state=0, inner_random_state=1):
    """
    dass_targets: dict like {
        "DASS21_depression": ("depression_group", "high"),
        "DASS21_anxiety": ("anxiety_group", "high"),
        "DASS21_stress": ("stress_group", "high"),
    }
    mapping subscale name -> (binary target column, positive class label)
    Assumes the binary group columns already exist in `dataset`,
    OR built from a cutoff -- adjust depending on your answer above.
    """
    embedding_cols = [c for c in dataset.columns if c.startswith(response_type) and "_embedding_" in c]
    pipelines_list = make_pipelines()

    dass_results = {}
    perm_results = {}

    for subscale, (target_col, positive_class) in dass_targets.items():
        print(f"\n{'='*60}\nDASS SUBSCALE: {subscale}\n{'='*60}")

        sub_dataset = dataset.dropna(subset=embedding_cols + [target_col]).reset_index(drop=True)
        X_data = sub_dataset[embedding_cols].to_numpy()
        y_data_sub = ((sub_dataset[target_col] == positive_class) * 1).to_numpy()

        print(f"N = {len(y_data_sub)}, positive proportion = {y_data_sub.mean():.3f}")

        results = run_nested_cv(
            X_data, y_data_sub, pipelines_list,
            n_outer_splits=n_outer_splits, n_inner_splits=n_inner_splits,
            outer_random_state=outer_random_state, inner_random_state=inner_random_state,
        )
        dass_results[subscale] = results

        fam, _ = pick_best_family(results)
        fam_pipelines = [p for p in pipelines_list if p[0] == fam]

        print(f"\n--- Permutation test: {subscale} (best family: {fam}) ---")
        obs, perm_scores, pval = permutation_test_nested_cv(
            X_data, y_data_sub, fam_pipelines, n_permutations=500
        )
        perm_results[subscale] = {"observed": obs, "perm_scores": perm_scores, "pvalue": pval, "family": fam}

    return dass_results, perm_results

def report_precision_recall(pooled, threshold=0.5, label=""):
    y_true = pooled["y_true"]
    y_score = pooled["y_score"]
    
    # note: SVM's y_score comes from decision_function, not predict_proba,
    # so 0.5 is NOT the right threshold for SVM -- 0 is (decision boundary)
    y_pred = (y_score >= threshold).astype(int)
    
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    acc = accuracy_score(y_true, y_pred)
    
    print(f"{label}: precision={precision:.3f}, recall={recall:.3f}, f1={f1:.3f}, accuracy={acc:.3f}")
    return precision, recall, f1, acc

def report_best_family_metrics(results, label=""):
    fam, _ = pick_best_family(results)
    pooled = results["per_family_pooled"][fam]
    threshold = 0.0 if fam == "svm" else 0.5
    precision, recall, f1, acc = report_precision_recall(pooled, threshold=threshold, label=f"{label} ({fam})")
    return fam, precision, recall, f1, acc

def run_cutoff_sensitivity(dataset, response_type, pcl_col="PCL-5_total",
                            cutoffs=(31, 33, 34, 36, 38),
                            n_outer_splits=5, n_inner_splits=5,
                            outer_random_state=0, inner_random_state=1):
    embedding_cols = [c for c in dataset.columns if c.startswith(response_type) and "_embedding_" in c]
    pipelines_list = make_pipelines()

    cutoff_results = {}

    for cutoff in cutoffs:
        print(f"\n{'='*60}\nCUTOFF: {cutoff}\n{'='*60}")

        sub_dataset = dataset.dropna(subset=embedding_cols + [pcl_col]).reset_index(drop=True)
        X_data = sub_dataset[embedding_cols].to_numpy()
        y_data_sub = (sub_dataset[pcl_col] >= cutoff).astype(int).to_numpy()

        print(f"N = {len(y_data_sub)}, positive proportion = {y_data_sub.mean():.3f}")

        results = run_nested_cv(
            X_data, y_data_sub, pipelines_list,
            n_outer_splits=n_outer_splits, n_inner_splits=n_inner_splits,
            outer_random_state=outer_random_state, inner_random_state=inner_random_state,
            verbose=False,   # keep this one quiet, we only need the summary
        )
        cutoff_results[cutoff] = results

        fam, scores = pick_best_family(results)
        print(f"Best family: {fam}, mean outer AUC = {scores.mean():.3f} ± {scores.std():.3f}, "
              f"95% CI [{results['auc_ci'][0]:.3f}, {results['auc_ci'][1]:.3f}]")

    return cutoff_results

def make_pipelines_ngram_fixed():
    pipelines_list = []

    for weighting in ["no_weight", "class_weight"]:
        cw = None if weighting == "no_weight" else "balanced"
        for ngram_range in [(1, 1), (1, 2)]:
            for C in [0.01, 0.1, 1.0, 10.0]:
                pipe = Pipeline([
                    ("vectorizer", TfidfVectorizer(
                        ngram_range=ngram_range, min_df=2, max_features=2000
                    )),
                    ("model", LogisticRegression(
                        solver="liblinear", C=C, class_weight=cw,
                        random_state=0, penalty="l2", max_iter=1000
                    ))
                ])
                pipelines_list.append(("ngram_logreg", f"{weighting}_{ngram_range}", C, pipe))

    for weighting in ["no_weight", "class_weight"]:
        cw = None if weighting == "no_weight" else "balanced"
        for ngram_range in [(1, 1), (1, 2)]:
            for C in [0.01, 0.1, 1.0, 10.0]:
                pipe = Pipeline([
                    ("vectorizer", TfidfVectorizer(
                        ngram_range=ngram_range, min_df=2, max_features=2000
                    )),
                    ("model", LinearSVC(C=C, class_weight=cw, random_state=0, max_iter=5000))
                ])
                pipelines_list.append(("ngram_svm", f"{weighting}_{ngram_range}", C, pipe))

    for ngram_range in [(1, 1), (1, 2)]:
        for n_estimators, max_depth in [(100, 2), (100, 3), (200, 2), (200, 3)]:
            pipe = Pipeline([
                ("vectorizer", TfidfVectorizer(
                    ngram_range=ngram_range, min_df=2, max_features=2000
                )),
                ("model", GradientBoostingClassifier(
                    n_estimators=n_estimators, max_depth=max_depth, random_state=0
                ))
            ])
            pipelines_list.append(("ngram_gbt", f"{ngram_range}_depth{max_depth}", n_estimators, pipe))

    return pipelines_list