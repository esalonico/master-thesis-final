"""
Fixed recall classifier evaluation with caching support.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, average_precision_score, confusion_matrix, f1_score, fbeta_score, precision_recall_curve, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from tqdm.auto import tqdm

from .classifier_cache import ClassifierCache
from .classifiers import FoldResult


@dataclass
class FixedRecallFoldResult(FoldResult):
    """
    Extended FoldResult that includes information about fixed recall evaluation.

    Additional attributes:
        target_recall (float): The target recall we aimed for.
        actual_recall (float): The actual recall achieved.
        threshold (float): The threshold used to achieve this recall.
        recall_achievable (bool): Whether the target recall was achievable.
    """

    target_recall: float
    actual_recall: float
    threshold: float
    recall_achievable: bool


def find_recall_threshold(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    target_recall: float = 0.95,
) -> Tuple[float, float, bool]:
    """
    Find the optimal classification threshold that achieves a target recall value (or gets as close as possible to it).

    Args:
        y_true: True binary labels
        y_scores: Target scores (probabilities or decision function values)
        target_recall: Target recall value (default 0.95)

    Returns:
        Tuple of (threshold, actual_recall, is_achievable)
        - threshold: The threshold value to use
        - actual_recall: The actual recall achieved with this threshold
        - is_achievable: Whether the exact target recall was achievable
    """

    # 1) Generate Precision-Recall Curve

    # get precision-recall curve (compute all possible precision/recall combinations across different thresholds)
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_scores)

    # note:
    # - recalls array is in decreasing order (high recall → low recall)
    # - thresholds array is in increasing order (low threshold → high threshold)
    # this inverse relationship exists because lower thresholds classify more samples as positive, increasing recall

    # 2) Handle Edge Cases

    # handle edge case where we have no positive samples
    if len(recalls) == 0 or np.sum(y_true) == 0:
        print("Warning: No positive samples in y_true or empty recalls array.")
        return 0.0, 0.0, False

    # 3) Find Best Threshold

    # find the threshold that gives us the target recall (or closest to it)
    # note: recalls are in decreasing order, thresholds in increasing order

    # calculates absolute difference between each possible recall and the target
    recall_diffs = np.abs(recalls - target_recall)

    # finds the index with minimum difference (closest match)
    best_idx = np.argmin(recall_diffs)

    # 4) Check Achievability

    # determine if the target recall is exactly achievable (within small tolerance)
    is_achievable = recall_diffs[best_idx] < 1e-6

    actual_recall = recalls[best_idx]

    # 5) Handle Boundary Case

    # handle the case where best_idx is the last element (no threshold available)
    if best_idx >= len(thresholds):
        print("Warning: best_idx >= len(thresholds)")
        # happens when best recall is at the end of the curve. In this case, use very low threshold (predict everything as positive)
        threshold = np.min(y_scores) - 1e-6
    else:
        threshold = thresholds[best_idx]

    return threshold, actual_recall, is_achievable


def evaluate_classifier_fixed_recall(
    X: np.ndarray,
    y: np.ndarray,
    classifier,
    target_recall: float = 0.95,
    random_state: int = 42,
    n_splits: int = 5,
    cache: Optional[ClassifierCache] = None,
    embedding_model: str = "default",
    classifier_name: str = "default",
) -> Tuple[List[FixedRecallFoldResult], Dict[str, float]]:
    """
    Cross-validated evaluation for a single classifier with fixed recall constraint.
    "How well does this classifier perform when I require it to achieve a specific recall (e.g., 95%)?"

    Unlike standard cross-validation that uses default thresholds (usually 0.5),
    this function optimizes the decision threshold in each fold to achieve a target recall,
    then measures all other performance metrics under that constraint.

    Args:
        X: Feature matrix
        y: Target labels
        classifier: Classifier to evaluate
        target_recall: Target recall to achieve (default 0.95)
        random_state: Random state for reproducibility
        n_splits: Number of cross-validation splits
        cache: Optional ClassifierCache instance for caching results
        embedding_model: Name of embedding model (for cache identification)
        classifier_name: Name of classifier (for cache identification)

    Returns:
        Tuple of (fold_results, aggregate_metrics)
    """
    # Check cache first if provided
    if cache is not None:
        cached_result = cache.get_cached_fixed_recall_result(
            X, y, classifier, embedding_model, classifier_name, target_recall, random_state, n_splits
        )
        if cached_result is not None:
            fold_results, agg_results, meta = cached_result
            print(f"Loaded cached fixed recall results for {embedding_model} + {classifier_name} (target_recall={target_recall})")
            return fold_results, agg_results

    # If we get here, we need to compute
    if cache is not None:
        print(f"Computing fixed recall results for {embedding_model} + {classifier_name} (target_recall={target_recall})")

    # set up cross-validation
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    fold_results: List[FixedRecallFoldResult] = []

    # cross-validation loop
    for fold_idx, (tr_idx, te_idx) in enumerate(skf.split(X, y)):
        X_tr, X_te = X[tr_idx], X[te_idx]
        y_tr, y_te = y[tr_idx], y[te_idx]

        # clone and fit the classifier (clone to ensure fresh model each fold)
        from sklearn.base import clone

        # train model
        model = clone(classifier)
        model.fit(X_tr, y_tr)

        # get prediction scores (continuous) for threshold tuning
        if hasattr(model, "predict_proba"):  # logistic regression and similar
            y_scores = model.predict_proba(X_te)[:, 1]

        elif hasattr(model, "decision_function"):  # SVMs and similar
            y_scores = model.decision_function(X_te)

        else:  # fallback: use predictions as scores (not ideal for threshold tuning)
            y_scores = model.predict(X_te).astype(float)

        # find the threshold for target recall: "what threshold gives us the desired recall?"
        threshold, actual_recall, recall_achievable = find_recall_threshold(y_te, y_scores, target_recall)

        # apply threshold to get binary predictions
        if hasattr(model, "predict_proba"):
            y_pred = (y_scores >= threshold).astype(int)
        elif hasattr(model, "decision_function"):
            y_pred = (y_scores >= threshold).astype(int)
        else:
            y_pred = (y_scores >= threshold).astype(int)

        # calculate all metrics with the threshold-adjusted predictions
        try:
            roc_auc = roc_auc_score(y_te, y_scores)  # threshold independent
        except ValueError:
            roc_auc = float("nan")

        pr_auc = average_precision_score(y_te, y_scores)
        f1 = f1_score(y_te, y_pred, zero_division=0)
        f2 = fbeta_score(y_te, y_pred, beta=2.0, zero_division=0)
        acc = accuracy_score(y_te, y_pred)

        # calculate confusion matrix and detailed metrics
        cm = confusion_matrix(y_te, y_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()

        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

        fold_results.append(
            FixedRecallFoldResult(
                roc_auc=roc_auc,
                pr_auc=pr_auc,
                f1=f1,
                f2=f2,
                acc=acc,
                recall=recall,
                precision=precision,
                specificity=specificity,
                support_pos=int((y_te == 1).sum()),
                support_neg=int((y_te == 0).sum()),
                target_recall=target_recall,
                actual_recall=actual_recall,
                threshold=threshold,
                recall_achievable=recall_achievable,
            )
        )

    # aggregate metrics over folds
    agg = {
        "roc_auc_mean": np.nanmean([f.roc_auc for f in fold_results]),
        "pr_auc_mean": np.nanmean([f.pr_auc for f in fold_results]),
        "f1_mean": np.nanmean([f.f1 for f in fold_results]),
        "f2_mean": np.nanmean([f.f2 for f in fold_results]),
        "acc_mean": np.nanmean([f.acc for f in fold_results]),
        "recall_mean": np.nanmean([f.recall for f in fold_results]),
        "precision_mean": np.nanmean([f.precision for f in fold_results]),
        "specificity_mean": np.nanmean([f.specificity for f in fold_results]),
        "target_recall": target_recall,
        "actual_recall_mean": np.nanmean([f.actual_recall for f in fold_results]),
        "threshold_mean": np.nanmean([f.threshold for f in fold_results]),
        "recall_achievable_ratio": np.mean([f.recall_achievable for f in fold_results]),
    }

    # Save to cache if provided
    if cache is not None:
        cache.save_fixed_recall_result(X, y, classifier, embedding_model, classifier_name, fold_results, agg, target_recall, random_state, n_splits)

    return fold_results, agg


def benchmark_classifiers_fixed_recall(
    X_by_model: Dict[str, np.ndarray],
    y: np.ndarray,
    classifiers: Dict[str, any],
    target_recall: float = 0.95,
    random_state: int = 42,
    n_splits: int = 5,
    cache: Optional[ClassifierCache] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Evaluate all (embedding model, classifier) combinations with fixed recall.

    Args:
        X_by_model: Dictionary mapping embedding model names to feature matrices
        y: Target labels
        classifiers: Dictionary mapping classifier names to classifier instances
        target_recall: Target recall to achieve (default 0.95)
        random_state: Random state for reproducibility
        n_splits: Number of cross-validation splits
        cache: Optional ClassifierCache instance for caching results

    Returns:
        Tuple of (per_fold_df, summary_df) - DataFrames with detailed and aggregated results
    """
    records_folds = []
    records_summary = []

    pbar_embedding_model = tqdm(list(X_by_model.items()), desc="Embedding models")
    for embed_name, X in pbar_embedding_model:
        pbar_embedding_model.set_description(f"Embedding: {embed_name}")

        pbar_classifier = tqdm(list(classifiers.items()), desc=f"Classifiers for {embed_name}", leave=False)
        for clf_name, clf in pbar_classifier:
            pbar_classifier.set_description(f"Classifier: {clf_name}")

            fold_results, agg = evaluate_classifier_fixed_recall(
                X,
                y,
                clf,
                target_recall=target_recall,
                random_state=random_state,
                n_splits=n_splits,
                cache=cache,
                embedding_model=embed_name,
                classifier_name=clf_name,
            )

            # Per-fold rows
            for i, fr in enumerate(fold_results, 1):
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
                        "target_recall": fr.target_recall,
                        "actual_recall": fr.actual_recall,
                        "threshold": fr.threshold,
                        "recall_achievable": fr.recall_achievable,
                    }
                )

            # Summary row
            rec = {"embedding_model": embed_name, "classifier": clf_name}
            rec.update(agg)
            records_summary.append(rec)

        pbar_classifier.close()
    pbar_embedding_model.close()

    per_fold_df = pd.DataFrame.from_records(records_folds).round(2)
    summary_df = (
        pd.DataFrame.from_records(records_summary).sort_values(["precision_mean", "f1_mean"], ascending=False).reset_index(drop=True).round(2)
    )

    return per_fold_df, summary_df


def analyze_recall_precision_tradeoff(
    X: np.ndarray,
    y: np.ndarray,
    classifier,
    recall_targets: List[float] = [0.9, 0.92, 0.95, 0.97, 0.99],
    random_state: int = 42,
    n_splits: int = 5,
    cache: Optional[ClassifierCache] = None,
    embedding_model: str = "default",
    classifier_name: str = "default",
) -> pd.DataFrame:
    """
    Analyze the precision-recall tradeoff for different target recall values.

    Args:
        X: Feature matrix
        y: Target labels
        classifier: Classifier to evaluate
        recall_targets: List of target recall values to test
        random_state: Random state for reproducibility
        n_splits: Number of cross-validation splits
        cache: Optional ClassifierCache instance for caching results
        embedding_model: Name of embedding model (for cache identification)
        classifier_name: Name of classifier (for cache identification)

    Returns:
        DataFrame with results for each recall target
    """
    results = []

    pbar_target_recall = tqdm(recall_targets, desc="Testing recall targets")
    for target_recall in pbar_target_recall:
        pbar_target_recall.set_description(f"Target Recall: {target_recall:.2f}")
        fold_results, agg = evaluate_classifier_fixed_recall(
            X,
            y,
            classifier,
            target_recall=target_recall,
            random_state=random_state,
            n_splits=n_splits,
            cache=cache,
            embedding_model=embedding_model,
            classifier_name=classifier_name,
        )

        result = {
            "target_recall": target_recall,
            "actual_recall_mean": agg["actual_recall_mean"],
            "precision_mean": agg["precision_mean"],
            "specificity_mean": agg["specificity_mean"],
            "f1_mean": agg["f1_mean"],
            "f2_mean": agg["f2_mean"],
            "accuracy_mean": agg["acc_mean"],
            "roc_auc_mean": agg["roc_auc_mean"],
            "pr_auc_mean": agg["pr_auc_mean"],
            "threshold_mean": agg["threshold_mean"],
            "recall_achievable_ratio": agg["recall_achievable_ratio"],
        }
        results.append(result)

    return pd.DataFrame(results)
