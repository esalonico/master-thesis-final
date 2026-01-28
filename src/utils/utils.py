"""
tiab_benchmark_utils.py

Utilities to benchmark embedding models (Hugging Face / sentence-transformers)
with classic ML classifiers for Title & Abstract (TIAB) screening.
"""

from __future__ import annotations

import hashlib
import html
import json
import math
import os
import re
import time
from dataclasses import asdict, dataclass

# sentence-transformers
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, average_precision_score, confusion_matrix, f1_score, fbeta_score, precision_recall_curve, roc_auc_score

# sklearn
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from tqdm import tqdm
from xgboost import XGBClassifier

# -----------------------------
# Data cleaning & preparation
# -----------------------------


def clean_html(text: str) -> str:
    """
    Remove HTML tags and decode HTML entities.
    """
    if pd.isna(text) or not isinstance(text, str):
        return str(text)

    # parse HTML and extract text
    soup = BeautifulSoup(text, "html.parser")
    clean_text = soup.get_text()

    # decode any remaining HTML entities
    clean_text = html.unescape(clean_text)

    # clean up whitespace
    clean_text = " ".join(clean_text.split())

    return clean_text


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Basic cleaning for TIAB dataset:
    - ensure cols exist
    - drop rows with no title
    - strip whitespace
    - remove html tags
    - drop duplicates (title+abstract)
    - ensure data types
    """
    assert "title" in df.columns
    assert "abstract" in df.columns
    assert "label" in df.columns

    df = df.copy()

    # remove rows with no title
    mask_nonempty = df["title"].str.len() > 0
    df = df.loc[mask_nonempty].copy()

    # strip
    df["title"] = df["title"].astype(str).str.strip()
    df["abstract"] = df["abstract"].astype(str).str.strip()

    # remove html tags from title and abstract
    df["title"] = df["title"].apply(clean_html)
    df["abstract"] = df["abstract"].apply(clean_html)

    # deduplicate by title + abstract
    df["__concat__"] = (df["title"] + " || " + df["abstract"]).str.lower().str.replace(r"\s+", " ", regex=True)
    df = df.drop_duplicates(subset="__concat__").drop(columns=["__concat__"])

    # ensure data types of labels
    df["label"] = df["label"].astype(bool)

    return df.reset_index(drop=True)


def join_title_abstract(df: pd.DataFrame, separator: str = " ") -> List[str]:
    """
    Join title and abstract in a single string for embedding.
    """
    return (df["title"] + separator + df["abstract"]).tolist()


def sha1_of_list_str(xs: List[str]) -> str:
    """
    Compute a deterministic SHA-1 hash for a list of strings.

    Each string is encoded in UTF-8 (ignoring errors) and separated by a newline
    before updating the hash. This produces a unique hash for the exact sequence
    and content of the list, suitable for caching or fingerprinting datasets.

    Args:
        xs (List[str]): List of strings to hash.

    Returns:
        str: SHA-1 hexadecimal digest representing the list.
    """
    h = hashlib.sha1()
    for s in xs:
        h.update(s.encode("utf-8", errors="ignore"))
        h.update(b"\n")
    return h.hexdigest()


# -----------------------------
# Metrics helpers
# -----------------------------


def wss_at_recall(y_true: np.ndarray, y_scores: np.ndarray, target_recall: float = 0.95) -> float:
    """
    Work Saved over Sampling at target recall (WSS@R).
    Sorts items by descending score; finds the smallest k achieving recall>=target.
    WSS = 1 - k/N.
    Returns 0 if target recall not achievable.
    """
    y_true = np.asarray(y_true).astype(int)
    y_scores = np.asarray(y_scores)

    n = len(y_true)
    order = np.argsort(-y_scores)  # desc
    y_sorted = y_true[order]

    total_pos = y_true.sum()
    if total_pos == 0:
        return 0.0

    cum_tp = np.cumsum(y_sorted)
    target_tp = math.ceil(target_recall * total_pos)

    idx = np.where(cum_tp >= target_tp)[0]
    if len(idx) == 0:
        return 0.0
    k = int(idx[0]) + 1  # number screened to reach target recall
    wss = 1.0 - (k / n)
    return max(0.0, min(1.0, float(wss)))


def threshold_for_target_recall(y_true: np.ndarray, y_scores: np.ndarray, target_recall: float = 0.95) -> float:
    """
    Find the lowest score threshold that achieves at least target_recall on y_true.
    Returns threshold value (float). If not achievable, returns +inf.
    """
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_scores)
    # thresholds len = len(precisions) - 1
    thr = np.inf
    for t, r in zip(thresholds, recalls[1:]):  # recalls aligned to thresholds
        if r >= target_recall:
            thr = min(thr, t)
    return thr


def sample_test_papers(
    df: pd.DataFrame,
    sample_size: int,
    included_percentage: float,
    label_col: str,
    random_state: int,
) -> pd.DataFrame:
    """
    Sample a small set of papers from the dataframe ensuring a requested number
    of included papers derived from `included_percentage`.
    """
    if included_percentage != -1:
        assert 0 <= included_percentage <= 1, "included_percentage must be between 0 and 1"

    included_df = df[df[label_col] == True]
    excluded_df = df[df[label_col] == False]

    # determine requested included count
    requested_included = int(math.ceil(sample_size * float(included_percentage))) if included_percentage != -1 else 0

    # clamp to availability
    n_included = max(0, min(requested_included, len(included_df)))

    # Sample included papers first
    included_sample = included_df.sample(n=n_included, random_state=random_state) if n_included > 0 else included_df.iloc[0:0]

    # Fill remaining slots from the rest of the dataset (excluding already selected)
    remaining = max(sample_size - n_included, 0)
    pool = df.drop(included_sample.index)

    remaining_sample = pool.sample(n=min(remaining, len(pool)), random_state=random_state) if remaining > 0 else pool.iloc[0:0]

    # Combine and shuffle
    sampled = pd.concat([included_sample, remaining_sample]).sample(frac=1, random_state=random_state).reset_index(drop=True)

    # Ensure we have at least one excluded paper (if possible)
    if sampled[label_col].all() and len(excluded_df) > 0:
        rep = excluded_df.sample(n=1, random_state=random_state)
        sampled.iloc[-1] = rep.iloc[0]

    # Final check: warn if not enough included papers available
    if sampled[label_col].sum() < requested_included:
        print(f"Warning: Could only sample {int(sampled[label_col].sum())} included papers (requested {requested_included}).")

    print(f"Selected {len(sampled)} sample papers: {sampled[label_col].sum()} included, {len(sampled)-sampled[label_col].sum()} excluded")
    return sampled
