from __future__ import annotations

import hashlib
import json
import os
import pickle
import re
import time
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np


def _compute_data_hash(X: np.ndarray, y: np.ndarray) -> str:
    """
    Compute a hash for the training data (X, y) to detect changes.
    """
    h = hashlib.sha1()
    # Hash the array data
    h.update(X.tobytes())
    h.update(y.tobytes())
    # Add shape information
    h.update(str(X.shape).encode())
    h.update(str(y.shape).encode())
    return h.hexdigest()


def _compute_classifier_hash(classifier, random_state: int, n_splits: int) -> str:
    """
    Compute a hash for classifier configuration to detect parameter changes.
    """
    h = hashlib.sha1()
    # Hash classifier type and parameters
    h.update(str(type(classifier)).encode())

    # Try to get parameters if available
    if hasattr(classifier, "get_params"):
        params = classifier.get_params()
        # Sort for consistency
        param_str = json.dumps(params, sort_keys=True, default=str)
        h.update(param_str.encode())

    # Add cross-validation parameters
    h.update(str(random_state).encode())
    h.update(str(n_splits).encode())

    return h.hexdigest()


def _compute_fixed_recall_classifier_hash(classifier, random_state: int, n_splits: int, target_recall: float) -> str:
    """
    Compute a hash for fixed recall classifier configuration to detect parameter changes.
    """
    h = hashlib.sha1()
    # Hash classifier type and parameters
    h.update(str(type(classifier)).encode())

    # Try to get parameters if available
    if hasattr(classifier, "get_params"):
        params = classifier.get_params()
        # Sort for consistency
        param_str = json.dumps(params, sort_keys=True, default=str)
        h.update(param_str.encode())

    # Add cross-validation parameters
    h.update(str(random_state).encode())
    h.update(str(n_splits).encode())

    # Add target recall for fixed recall classifiers
    h.update(str(target_recall).encode())

    return h.hexdigest()


@dataclass
class ClassifierMeta:
    embedding_model: str
    classifier_name: str
    classifier_type: str
    data_hash: str
    classifier_hash: str
    n_samples: int
    n_features: int
    n_splits: int
    random_state: int
    created_at: float
    classifier_params: Optional[Dict] = None


@dataclass
class FixedRecallClassifierMeta(ClassifierMeta):
    target_recall: float = None


class ClassifierCache:
    """
    Cache trained classifier evaluation results to disk to avoid recomputing.
    Similar to EmbeddingCache but for classifier evaluation results.
    """

    def __init__(self, cache_dir: str = "./classifier_cache"):
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)

    def _paths(self, embedding_model: str, classifier_name: str, data_hash: str, classifier_hash: str) -> Tuple[str, str, str]:
        """Generate file paths for cached classifier results."""
        # Create safe directory names
        safe_embed = re.sub(r"[^a-zA-Z0-9._-]+", "_", embedding_model)
        safe_clf = re.sub(r"[^a-zA-Z0-9._-]+", "_", classifier_name)

        # Create subdirectory structure: cache_dir/EMBEDDING_MODEL/CLASSIFIER_NAME/
        subdir = os.path.join(self.cache_dir, safe_embed, safe_clf)
        os.makedirs(subdir, exist_ok=True)

        # Simplified filename with only hashes
        base = f"{data_hash[:8]}__{classifier_hash[:8]}"

        results_path = os.path.join(subdir, base + "_results.pkl")
        summary_path = os.path.join(subdir, base + "_summary.json")
        meta_path = os.path.join(subdir, base + "_meta.json")

        return results_path, summary_path, meta_path

    def _save(self, fold_results: List, agg_results: Dict, meta: ClassifierMeta, results_path: str, summary_path: str, meta_path: str):
        """Save classifier evaluation results to disk."""
        # Save fold results as pickle (contains FoldResult objects)
        with open(results_path, "wb") as f:
            pickle.dump(fold_results, f)

        # Custom JSON serializer that handles NaN, inf, and other problematic values
        def json_serializer(obj):
            if isinstance(obj, (np.float32, np.float64)):
                if np.isnan(obj):
                    return None
                elif np.isinf(obj):
                    return "inf" if obj > 0 else "-inf"
                else:
                    return float(obj)
            elif isinstance(obj, (np.int32, np.int64)):
                return int(obj)
            elif hasattr(obj, "__dict__"):
                return str(obj)
            else:
                return str(obj)

        # Save summary results as JSON with safe serialization
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(agg_results, f, indent=2, default=json_serializer)

        # Save metadata as JSON with safe serialization
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(asdict(meta), f, indent=2, default=json_serializer)

    def _load(self, results_path: str, summary_path: str, meta_path: str) -> Tuple[List, Dict, Dict]:
        """Load cached classifier evaluation results from disk."""
        if not (os.path.exists(results_path) and os.path.exists(summary_path) and os.path.exists(meta_path)):
            raise FileNotFoundError("Cache files not found")

        # Load fold results
        with open(results_path, "rb") as f:
            fold_results = pickle.load(f)

        # Load summary results
        with open(summary_path, "r", encoding="utf-8") as f:
            agg_results = json.load(f)

        # Load metadata as dict (don't convert to dataclass here)
        with open(meta_path, "r", encoding="utf-8") as f:
            meta_dict = json.load(f)

        return fold_results, agg_results, meta_dict

    def get_cached_result(
        self, X: np.ndarray, y: np.ndarray, classifier, embedding_model: str, classifier_name: str, random_state: int = 42, n_splits: int = 5
    ) -> Optional[Tuple[List, Dict, ClassifierMeta]]:
        """
        Try to load cached results for the given configuration.
        Returns None if no valid cache is found.
        """
        try:
            data_hash = _compute_data_hash(X, y)
            classifier_hash = _compute_classifier_hash(classifier, random_state, n_splits)

            results_path, summary_path, meta_path = self._paths(embedding_model, classifier_name, data_hash, classifier_hash)

            fold_results, agg_results, meta_dict = self._load(results_path, summary_path, meta_path)

            # Convert dict to ClassifierMeta
            if "target_recall" in meta_dict:
                # Remove target_recall for regular ClassifierMeta
                meta_dict_clean = {k: v for k, v in meta_dict.items() if k != "target_recall"}
                meta = ClassifierMeta(**meta_dict_clean)
            else:
                meta = ClassifierMeta(**meta_dict)

            # Verify the cache is still valid
            if (
                meta.data_hash == data_hash
                and meta.classifier_hash == classifier_hash
                and meta.embedding_model == embedding_model
                and meta.classifier_name == classifier_name
                and meta.random_state == random_state
                and meta.n_splits == n_splits
            ):
                return fold_results, agg_results, meta
            else:
                print(f"Cache invalid for {embedding_model} + {classifier_name}: metadata mismatch")
                return None

        except Exception as e:
            # Cache miss or error
            return None

    def save_result(
        self,
        X: np.ndarray,
        y: np.ndarray,
        classifier,
        embedding_model: str,
        classifier_name: str,
        fold_results: List,
        agg_results: Dict,
        random_state: int = 42,
        n_splits: int = 5,
    ):
        """
        Save classifier evaluation results to cache.
        """
        try:
            data_hash = _compute_data_hash(X, y)
            classifier_hash = _compute_classifier_hash(classifier, random_state, n_splits)

            # Extract classifier parameters if available
            classifier_params = None
            if hasattr(classifier, "get_params"):
                try:
                    classifier_params = classifier.get_params()
                except Exception:
                    # Some classifiers might have issues with get_params()
                    classifier_params = None

            meta = ClassifierMeta(
                embedding_model=embedding_model,
                classifier_name=classifier_name,
                classifier_type=str(type(classifier)),
                data_hash=data_hash,
                classifier_hash=classifier_hash,
                n_samples=X.shape[0],
                n_features=X.shape[1],
                n_splits=n_splits,
                random_state=random_state,
                created_at=time.time(),
                classifier_params=classifier_params,
            )

            results_path, summary_path, meta_path = self._paths(embedding_model, classifier_name, data_hash, classifier_hash)

            self._save(fold_results, agg_results, meta, results_path, summary_path, meta_path)
            print(f"Cached results for {embedding_model} + {classifier_name}")

        except Exception as e:
            print(f"Failed to cache results for {embedding_model} + {classifier_name}: {e}")

    def get_cached_fixed_recall_result(
        self,
        X: np.ndarray,
        y: np.ndarray,
        classifier,
        embedding_model: str,
        classifier_name: str,
        target_recall: float,
        random_state: int = 42,
        n_splits: int = 5,
    ) -> Optional[Tuple[List, Dict, FixedRecallClassifierMeta]]:
        """
        Try to load cached results for the given fixed recall configuration.
        Returns None if no valid cache is found.
        """
        try:
            data_hash = _compute_data_hash(X, y)
            classifier_hash = _compute_fixed_recall_classifier_hash(classifier, random_state, n_splits, target_recall)

            results_path, summary_path, meta_path = self._paths(embedding_model, classifier_name, data_hash, classifier_hash)

            # Check if files exist
            if not (os.path.exists(results_path) and os.path.exists(summary_path) and os.path.exists(meta_path)):
                return None

            fold_results, agg_results, meta_dict = self._load(results_path, summary_path, meta_path)

            # Convert dict to FixedRecallClassifierMeta
            if "target_recall" not in meta_dict:
                meta_dict["target_recall"] = target_recall

            meta = FixedRecallClassifierMeta(**meta_dict)

            # Verify the cache is still valid
            if (
                meta.data_hash == data_hash
                and meta.classifier_hash == classifier_hash
                and meta.embedding_model == embedding_model
                and meta.classifier_name == classifier_name
                and meta.random_state == random_state
                and meta.n_splits == n_splits
                and meta.target_recall == target_recall
            ):
                return fold_results, agg_results, meta
            else:
                print(f"Cache invalid for {embedding_model} + {classifier_name} (target_recall={target_recall}): metadata mismatch")
                return None

        except Exception as e:
            # Cache miss or error
            return None

    def save_fixed_recall_result(
        self,
        X: np.ndarray,
        y: np.ndarray,
        classifier,
        embedding_model: str,
        classifier_name: str,
        fold_results: List,
        agg_results: Dict,
        target_recall: float,
        random_state: int = 42,
        n_splits: int = 5,
    ):
        """
        Save fixed recall classifier evaluation results to cache.
        """
        try:
            data_hash = _compute_data_hash(X, y)
            classifier_hash = _compute_fixed_recall_classifier_hash(classifier, random_state, n_splits, target_recall)

            # Extract classifier parameters if available
            classifier_params = None
            if hasattr(classifier, "get_params"):
                try:
                    classifier_params = classifier.get_params()
                except Exception:
                    # Some classifiers might have issues with get_params()
                    classifier_params = None

            meta = FixedRecallClassifierMeta(
                embedding_model=embedding_model,
                classifier_name=classifier_name,
                classifier_type=str(type(classifier)),
                data_hash=data_hash,
                classifier_hash=classifier_hash,
                n_samples=X.shape[0],
                n_features=X.shape[1],
                n_splits=n_splits,
                random_state=random_state,
                created_at=time.time(),
                classifier_params=classifier_params,
                target_recall=target_recall,
            )

            results_path, summary_path, meta_path = self._paths(embedding_model, classifier_name, data_hash, classifier_hash)

            self._save(fold_results, agg_results, meta, results_path, summary_path, meta_path)
            print(f"Cached fixed recall results for {embedding_model} + {classifier_name} (target_recall={target_recall})")

        except Exception as e:
            print(f"Failed to cache fixed recall results for {embedding_model} + {classifier_name}: {e}")

    def clear_cache(self):
        """Remove all cached classifier results."""
        import shutil

        if os.path.exists(self.cache_dir):
            shutil.rmtree(self.cache_dir)
            os.makedirs(self.cache_dir, exist_ok=True)
            print(f"Cleared classifier cache at {self.cache_dir}")

    def list_cached_results(self) -> List[Dict]:
        """List all cached classifier results with their metadata."""
        cached_results = []

        if not os.path.exists(self.cache_dir):
            return cached_results

        # Walk through the directory structure: cache_dir/EMBEDDING_MODEL/CLASSIFIER_NAME/
        for embedding_dir in os.listdir(self.cache_dir):
            embedding_path = os.path.join(self.cache_dir, embedding_dir)
            if not os.path.isdir(embedding_path):
                continue

            for classifier_dir in os.listdir(embedding_path):
                classifier_path = os.path.join(embedding_path, classifier_dir)
                if not os.path.isdir(classifier_path):
                    continue

                for filename in os.listdir(classifier_path):
                    if filename.endswith("_meta.json"):
                        meta_path = os.path.join(classifier_path, filename)
                        try:
                            with open(meta_path, "r", encoding="utf-8") as f:
                                meta_dict = json.load(f)
                            cached_results.append(meta_dict)
                        except Exception:
                            continue

        return cached_results

    def get_all_cached_results(self) -> List[Dict]:
        """
        Get all cached results including fold_results data from the cache directory.
        
        Returns:
            List of dictionaries containing cached result information including fold_results
        """
        results = []
        
        if not os.path.exists(self.cache_dir):
            return results
        
        # Walk through the directory structure: cache_dir/EMBEDDING_MODEL/CLASSIFIER_NAME/
        for embedding_dir in os.listdir(self.cache_dir):
            embedding_path = os.path.join(self.cache_dir, embedding_dir)
            if not os.path.isdir(embedding_path):
                continue

            for classifier_dir in os.listdir(embedding_path):
                classifier_path = os.path.join(embedding_path, classifier_dir)
                if not os.path.isdir(classifier_path):
                    continue

                for filename in os.listdir(classifier_path):
                    if filename.endswith("_results.pkl"):
                        # Load the actual results file
                        results_path = os.path.join(classifier_path, filename)
                        meta_path = results_path.replace("_results.pkl", "_meta.json")
                        summary_path = results_path.replace("_results.pkl", "_summary.json")
                        
                        try:
                            # Load the pickle file with fold_results (it's a list of FoldResult objects)
                            with open(results_path, 'rb') as f:
                                fold_results = pickle.load(f)  # This is directly the list of FoldResult objects
                            
                            # Load metadata
                            meta = {}
                            if os.path.exists(meta_path):
                                with open(meta_path, "r", encoding="utf-8") as f:
                                    meta = json.load(f)
                            
                            # Load summary
                            agg = {}
                            if os.path.exists(summary_path):
                                with open(summary_path, "r", encoding="utf-8") as f:
                                    agg = json.load(f)
                            
                            results.append({
                                'filename': filename,
                                'embedding_model': meta.get('embedding_model', 'unknown'),
                                'classifier_name': meta.get('classifier_name', 'unknown'),
                                'fold_results': fold_results,
                                'agg': agg,
                                'meta': meta
                            })
                            
                        except Exception as e:
                            print(f"Warning: Could not load cache file {filename}: {e}")
                            continue
        
        return results
