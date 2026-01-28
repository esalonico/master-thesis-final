from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, average_precision_score, confusion_matrix, f1_score, fbeta_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from tqdm.auto import tqdm
from xgboost import XGBClassifier

from .classifier_cache import ClassifierCache


def make_logreg_classifier(random_state: int) -> Pipeline:
    """
    Logistic Regression (balanced) with StandardScaler.
    """
    return Pipeline(
        [
            ("scaler", StandardScaler(with_mean=True)),
            (
                "clf",
                LogisticRegression(
                    solver="lbfgs",  # lbfgs is better for small datasets (SAGA solver with only 500 iterations may not be enough for high-dimensional embeddings)
                    max_iter=500,
                    n_jobs=-1,
                    class_weight="balanced",  # handle class imbalance
                    verbose=0,
                    random_state=random_state,
                ),
            ),
        ]
    )


def make_svm_classifier() -> CalibratedClassifierCV:
    """
    Linear SVM with probability calibration.
    """
    return CalibratedClassifierCV(
        estimator=LinearSVC(class_weight="balanced"),
        method="sigmoid",
        cv=3,
    )


def make_xgb_classifier() -> XGBClassifier:
    """
    XGBoost classifier
    """
    return XGBClassifier(
        max_depth=6,
        n_estimators=400,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=1.0,  # let class_weight above handle imbalance if needed
        reg_lambda=1.0,
        n_jobs=4,
        tree_method="hist",
        objective="binary:logistic",
        eval_metric="logloss",
    )


def make_classifiers(random_state: int) -> Dict[str, any]:
    """
    Classic ML classifiers suited for dense embeddings.
    - Logistic Regression (balanced)
    - Linear SVM with probability calibration
    - XGBoost
    """
    clfs = {}

    # Logistic Regression
    clfs["LogReg"] = make_logreg_classifier(random_state)

    # Linear SVM (with calibration for probabilities)
    clfs["LinearSVM"] = make_svm_classifier()

    # XGBoost
    clfs["XGBoost"] = make_xgb_classifier()

    return clfs


@dataclass
class FoldResult:
    """
    Dataclass representing the results of a single fold in a model evaluation.

    Attributes:
        roc_auc (float): Receiver Operating Characteristic Area Under Curve score (https://developers.google.com/machine-learning/crash-course/classification/roc-and-auc)
        pr_auc (float): Precision-Recall Area Under Curve score.
        f1 (float): F1 score, the harmonic mean of precision and recall.
        f2 (float): F2 score, a weighted harmonic mean favoring recall.
        acc (float): Accuracy score, the proportion of correct predictions.
        recall (float): Recall score, the proportion of actual positives correctly identified.
        precision (float): Precision score, the proportion of positive identifications that were correct.
        specificity (float): Specificity score, the proportion of actual negatives correctly identified (TNR).
        support_pos (int): Number of positive samples in the fold.
        support_neg (int): Number of negative samples in the fold.
    """

    roc_auc: float
    pr_auc: float
    f1: float
    f2: float
    acc: float
    recall: float
    precision: float
    specificity: float
    support_pos: int
    support_neg: int


def evaluate_classifier(
    X: np.ndarray,
    y: np.ndarray,
    classfier,
    embedding_model: str = "unknown",
    classifier_name: str = "unknown",
    random_state: int = 42,
    n_splits: int = 5,
    cache: ClassifierCache = None,
) -> Tuple[List[FoldResult], Dict[str, float]]:
    """
    Cross-validated evaluation for a single classifier with caching support.
    Returns per-fold results and aggregate metrics (mean).

    Args:
        X: Feature matrix
        y: Target labels
        classfier: Classifier to evaluate
        embedding_model: Name of the embedding model (for caching)
        classifier_name: Name of the classifier (for caching)
        random_state: Random state for reproducibility
        n_splits: Number of cross-validation splits
        cache: ClassifierCache instance for caching results
    """
    # Try to load from cache first
    if cache is not None:
        cached_result = cache.get_cached_result(X, y, classfier, embedding_model, classifier_name, random_state, n_splits)
        if cached_result is not None:
            fold_results, agg, meta = cached_result
            print(f"Loaded cached results for {embedding_model} + {classifier_name}")
            return fold_results, agg

    # Compute results if not cached
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    fold_results: List[FoldResult] = []
    for tr_idx, te_idx in skf.split(X, y):
        X_tr, X_te = X[tr_idx], X[te_idx]
        y_tr, y_te = y[tr_idx], y[te_idx]

        model = classfier
        # Some pipelines need refit per fold (always clone-like behavior expected)
        # We'll fit directly; for Pipeline it's okay.
        model.fit(X_tr, y_tr)

        if hasattr(model, "predict_proba"):
            y_scores = model.predict_proba(X_te)[:, 1]
        elif hasattr(model, "decision_function"):
            y_scores = model.decision_function(X_te)
            # Map decision_function to [0,1] via logistic if desired; keep raw scores for ranking.
            # For AP/ROC, raw scores are fine.
        else:
            # Fallback: use predictions as scores
            y_scores = model.predict(X_te)

        # Default threshold 0.5 for base metrics if we have probabilities; otherwise sign(0).
        if y_scores.min() >= 0 and y_scores.max() <= 1:
            y_pred = (y_scores >= 0.5).astype(int)
        else:
            # zero threshold for margins
            y_pred = (y_scores >= 0.0).astype(int)

        # Metrics
        try:
            roc = roc_auc_score(y_te, y_scores)
        except ValueError:
            roc = float("nan")
        pr = average_precision_score(y_te, y_scores)
        f1 = f1_score(y_te, y_pred, zero_division=0)
        f2 = fbeta_score(y_te, y_pred, beta=2.0, zero_division=0)
        acc = accuracy_score(y_te, y_pred)
        cm = confusion_matrix(y_te, y_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

        fold_results.append(
            FoldResult(
                roc_auc=roc,
                pr_auc=pr,
                f1=f1,
                f2=f2,
                acc=acc,
                recall=recall,
                precision=precision,
                specificity=specificity,
                support_pos=int((y_te == 1).sum()),
                support_neg=int((y_te == 0).sum()),
            )
        )

    # Aggregate means
    agg = {
        "roc_auc_mean": np.nanmean([f.roc_auc for f in fold_results]),
        "pr_auc_mean": np.nanmean([f.pr_auc for f in fold_results]),
        "f1_mean": np.nanmean([f.f1 for f in fold_results]),
        "f2_mean": np.nanmean([f.f2 for f in fold_results]),
        "acc_mean": np.nanmean([f.acc for f in fold_results]),
        "recall_mean": np.nanmean([f.recall for f in fold_results]),
        "precision_mean": np.nanmean([f.precision for f in fold_results]),
        "specificity_mean": np.nanmean([f.specificity for f in fold_results]),
    }

    # Save to cache if provided
    if cache is not None:
        cache.save_result(X, y, classfier, embedding_model, classifier_name, fold_results, agg, random_state, n_splits)

    return fold_results, agg


def benchmark_classifiers(
    X_by_model: Dict[str, np.ndarray],
    y: np.ndarray,
    classifiers: Dict[str, any],
    random_state: int = 42,
    n_splits: int = 5,
    use_cache: bool = True,
    cache_dir: str = "./classifiers_cache",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Evaluate all (embedding model, classifier) combinations with caching support.
    Returns:
      - per_fold_df: one row per (embed, clf, fold)
      - summary_df: aggregated means per (embed, clf)

    Args:
        X_by_model: Dictionary mapping embedding model names to feature matrices
        y: Target labels
        classifiers: Dictionary mapping classifier names to classifier instances
        random_state: Random state for reproducibility
        n_splits: Number of cross-validation splits
        use_cache: Whether to use caching for classifier results
        cache_dir: Directory to store cache files
    """
    # Initialize cache if requested
    cache = ClassifierCache(cache_dir) if use_cache else None

    records_folds = []
    records_summary = []

    pbar_embedding_model = tqdm(list(X_by_model.items()), desc="Embedding models")
    for embed_name, X in pbar_embedding_model:
        pbar_embedding_model.set_description(f"Embedding: {embed_name}")

        pbar_classifier = tqdm(list(classifiers.items()), desc=f"Classifiers for {embed_name}", leave=False)
        for clf_name, clf in pbar_classifier:
            # update inner progress description to show current classifier name
            pbar_classifier.set_description(f"Classifier: {clf_name}")

            fold_results, agg = evaluate_classifier(
                X,
                y,
                clf,
                embedding_model=embed_name,
                classifier_name=clf_name,
                random_state=random_state,
                n_splits=n_splits,
                cache=cache,
            )

            # Per-fold rows
            pbar_folds = tqdm(range(1, len(fold_results) + 1), desc=f"{embed_name} | {clf_name} folds", leave=False)
            for i in pbar_folds:
                fr = fold_results[i - 1]
                pbar_folds.set_description(f"Fold {i}/{len(fold_results)}")
                records_folds.append(
                    {
                        "embedding_model": embed_name,
                        "classifier": clf_name,
                        "fold": i,
                        "roc_auc": fr.roc_auc,
                        "pr_auc": fr.pr_auc,
                        "f1": fr.f1,
                        "f2": fr.f2,
                        "accuracy": fr.acc,
                        "recall": fr.recall,
                        "precision": fr.precision,
                        "specificity": fr.specificity,
                        "support_pos": fr.support_pos,
                        "support_neg": fr.support_neg,
                    }
                )
            pbar_folds.close()

            # Summary row
            rec = {"embedding_model": embed_name, "classifier": clf_name}
            rec.update(agg)
            records_summary.append(rec)

        pbar_classifier.close()
    pbar_embedding_model.close()

    per_fold_df = pd.DataFrame.from_records(records_folds)
    summary_df = pd.DataFrame.from_records(records_summary).sort_values(["pr_auc_mean", "roc_auc_mean"], ascending=False).reset_index(drop=True)

    return per_fold_df, summary_df
