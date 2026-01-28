from __future__ import annotations

import json
import os
import re
import time
from dataclasses import asdict, dataclass
from typing import List, Optional, Tuple

import numpy as np
import openai
from adapters import AutoAdapterModel
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer

from utils.utils import sha1_of_list_str

load_dotenv()


@dataclass
class EmbeddingMeta:
    model_name: str
    n_items: int
    dim: int
    normalized: bool
    created_at: float
    dataset_hash: str


class EmbeddingCache:
    """
    Compute & cache embeddings to disk so we don't recompute every run.
    Uses SentenceTransformer under the hood.
    """

    def __init__(self, cache_dir: str = "./embeddings_cache"):
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)

    def _paths(self, model_name: str, dataset_hash: str) -> Tuple[str, str]:
        safe = re.sub(r"[^a-zA-Z0-9._-]+", "_", model_name)
        base = f"{safe}__{dataset_hash}"
        arr_path = os.path.join(self.cache_dir, base + ".npy")
        meta_path = os.path.join(self.cache_dir, base + ".json")
        return arr_path, meta_path

    def _save(self, arr: np.ndarray, meta: EmbeddingMeta, arr_path: str, meta_path: str):
        np.save(arr_path, arr)
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(asdict(meta), f, indent=2)

    def _load(self, arr_path: str, meta_path: str) -> Tuple[np.ndarray, EmbeddingMeta]:
        if not (os.path.exists(arr_path) and os.path.exists(meta_path)):
            raise FileNotFoundError
        arr = np.load(arr_path)
        with open(meta_path, "r", encoding="utf-8") as f:
            md = json.load(f)
        meta = EmbeddingMeta(**md)
        return arr, meta

    @staticmethod
    def is_specter2_model(model_name: str) -> bool:
        """Check if the model is a SPECTER2 model that requires adapter handling."""
        return "specter2" in model_name.lower()

    @staticmethod
    def is_openai_model(model_name: str) -> bool:
        """Check if the model is an OpenAI model."""
        return "text-embedding-" in model_name.lower()

    def compute(
        self,
        texts: List[str],
        model_name: str,
        batch_size: int = 64,
        normalize_embeddings: bool = False,
        device: Optional[str] = None,
        show_progress_bar: bool = True,
    ) -> Tuple[np.ndarray, EmbeddingMeta]:
        """
        Computes and returns embeddings for a list of input texts using a specified SentenceTransformer model.
        Embeddings are cached to disk and loaded if available; otherwise, they are computed and saved for future use.

        Args:
            texts (List[str]): List of input text strings to embed.
            model_name (str): Name or path of the SentenceTransformer model to use.
            batch_size (int, optional): Number of texts to process per batch. Defaults to 64.
            normalize_embeddings (bool, optional): Whether to normalize output embeddings. Defaults to False.
            device (Optional[str], optional): Device to run the model on (e.g., 'cpu', 'cuda'). Defaults to None.
            show_progress_bar (bool, optional): Whether to display a progress bar during embedding computation. Defaults to True.

        Returns:
            Tuple[np.ndarray, EmbeddingMeta]:
                - np.ndarray: Array of computed embeddings with shape (n_items, embedding_dim).
                - EmbeddingMeta: Metadata about the embeddings (model name, number of items, dimension, creation time, dataset hash).

        """
        dataset_hash = sha1_of_list_str(texts)
        arr_path, meta_path = self._paths(model_name, dataset_hash)

        # try load
        try:
            arr, meta = self._load(arr_path, meta_path)
            return arr, meta
        except Exception:
            pass

        # compute
        model = SentenceTransformer(model_name, device=device)
        emb = model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=normalize_embeddings,
            show_progress_bar=show_progress_bar,
        )
        emb = np.asarray(emb)

        # metadata
        meta = EmbeddingMeta(
            model_name=model_name,
            n_items=len(texts),
            dim=int(emb.shape[1]),
            normalized=normalize_embeddings,
            created_at=time.time(),
            dataset_hash=dataset_hash,
        )
        self._save(emb, meta, arr_path, meta_path)
        return emb, meta

    def normalize(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Normalize embeddings to unit length (L2 normalization).

        Args:
            embeddings (np.ndarray): Input embeddings to normalize.
        Returns:
            np.ndarray: L2-normalized embeddings.
        """
        # compute L2 normalization
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)

        # avoid division by zero
        norms = np.where(norms == 0, 1, norms)

        # compute normalized embeddings
        normalized_embeddings = embeddings / norms

        return normalized_embeddings


class SPECTER2EmbeddingCache(EmbeddingCache):
    """
    Specialized embedding cache for SPECTER2 models that handles adapter merging.
    This class automatically manages the adapter loading and merging process for SPECTER2 models.
    """

    def __init__(self, cache_dir: str = "./embeddings_cache", specter_cache_dir: str = "specter"):
        super().__init__(cache_dir)
        self.specter_cache_dir = os.path.join(cache_dir, specter_cache_dir)
        os.makedirs(self.specter_cache_dir, exist_ok=True)
        self._original_model_name = None  # Track original model name

    def _paths(self, model_name: str, dataset_hash: str) -> Tuple[str, str]:
        """Override to use original model name for cache file naming."""
        # Use original model name if we're processing a merged model path
        name_for_cache = self._original_model_name if self._original_model_name else model_name
        safe = re.sub(r"[^a-zA-Z0-9._-]+", "_", name_for_cache)
        base = f"{safe}__{dataset_hash}"
        arr_path = os.path.join(self.cache_dir, base + ".npy")
        meta_path = os.path.join(self.cache_dir, base + ".json")
        return arr_path, meta_path

    def _get_merged_model_path(self, model_name: str) -> str:
        """Generate path for the merged SPECTER2 model."""
        safe_name = re.sub(r"[^a-zA-Z0-9._-]+", "_", model_name)
        return os.path.join(self.specter_cache_dir, f"specter2_merged_{safe_name}")

    def _merge_specter2_adapters(self, model_name: str) -> str:
        """
        Merge SPECTER2 adapters and save the merged model.

        Args:
            model_name (str): The SPECTER2 model name (e.g., 'allenai/specter2_base')

        Returns:
            str: Path to the merged model directory
        """
        merged_model_path = self._get_merged_model_path(model_name)

        # check if merged model already exists
        if os.path.exists(merged_model_path) and os.path.exists(os.path.join(merged_model_path, "config.json")):
            # print(f"Using cached merged SPECTER2 model from: {merged_model_path}")
            return merged_model_path

        print(f"Merging SPECTER2 adapters for {model_name}...")

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.model_max_length = 512

        # Load base model with adapters
        model = AutoAdapterModel.from_pretrained(model_name)

        # Load and activate the adapter
        adapter_name = "allenai/specter2"
        model.load_adapter(adapter_name, source="hf", load_as="specter2", set_active=True)

        # Merge the adapter
        model.merge_adapter("specter2")

        # Save merged model and tokenizer
        model.save_pretrained(merged_model_path)
        tokenizer.save_pretrained(merged_model_path)

        print(f"SPECTER2 adapters merged and saved to: {merged_model_path}")
        return merged_model_path

    def compute(
        self,
        texts: List[str],
        model_name: str,
        batch_size: int = 64,
        normalize_embeddings: bool = False,
        device: Optional[str] = None,
        show_progress_bar: bool = True,
    ) -> Tuple[np.ndarray, EmbeddingMeta]:
        """
        Computes embeddings for SPECTER2 models with automatic adapter merging.
        For non-SPECTER2 models, falls back to the parent class implementation.

        Args:
            texts (List[str]): List of input text strings to embed.
            model_name (str): Name or path of the model to use.
            batch_size (int, optional): Number of texts to process per batch. Defaults to 64.
            normalize_embeddings (bool, optional): Whether to normalize output embeddings. Defaults to False.
            device (Optional[str], optional): Device to run the model on (e.g., 'cpu', 'cuda'). Defaults to None.
            show_progress_bar (bool, optional): Whether to display a progress bar during embedding computation. Defaults to True.

        Returns:
            Tuple[np.ndarray, EmbeddingMeta]: Array of computed embeddings and metadata.
        """
        # check if this is a SPECTER2 model
        if not self.is_specter2_model(model_name):
            raise ValueError(f"Model {model_name} is not a SPECTER2 model.")

        # Store original model name for cache file naming
        self._original_model_name = model_name

        try:
            # for SPECTER2 models, merge adapters first
            merged_model_path = self._merge_specter2_adapters(model_name)

            # use the merged model path for embedding computation
            result = super().compute(texts, merged_model_path, batch_size, normalize_embeddings, device, show_progress_bar)

            # Update metadata to reflect original model name
            embeddings, meta = result
            meta.model_name = model_name  # Use original name in metadata

            return embeddings, meta
        finally:
            # Reset original model name
            self._original_model_name = None


class OpenAIEmbeddingCache(EmbeddingCache):
    """
    OpenAI API-based embedding cache that uses OpenAI's embedding models.
    This class provides a consistent interface with local embedding models while using OpenAI's API.
    """

    def __init__(self, cache_dir: str = "./embeddings_cache", api_key: Optional[str] = None):
        """
        Initialize the OpenAI embedding cache.

        Args:
            cache_dir (str): Directory to store cached embeddings.
            api_key (Optional[str]): OpenAI API key. If None, reads from OPENAI_API_KEY environment variable.
        """
        super().__init__(cache_dir)
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key not found. Please set OPENAI_API_KEY environment variable.")
        self.client = openai.OpenAI(api_key=self.api_key)

    def compute(
        self,
        texts: List[str],
        model_name: str = "text-embedding-3-small",
        batch_size: int = 100,
        normalize_embeddings: bool = False,
        device: Optional[str] = None,  # not used for OpenAI API but kept for interface consistency
        show_progress_bar: bool = True,
    ) -> Tuple[np.ndarray, EmbeddingMeta]:
        """
        Computes and returns embeddings for a list of input texts using OpenAI's embedding API.
        Embeddings are cached to disk and loaded if available; otherwise, they are computed and saved for future use.

        Args:
            texts (List[str]): List of input text strings to embed.
            model_name (str): OpenAI embedding model to use (e.g., 'text-embedding-3-small', 'text-embedding-3-large').
            batch_size (int, optional): Number of texts to process per batch. Defaults to 100.
            normalize_embeddings (bool, optional): Whether to normalize output embeddings. Defaults to False.
            device (Optional[str], optional): Not used for OpenAI API but kept for interface consistency.
            show_progress_bar (bool, optional): Whether to display a progress bar during embedding computation. Defaults to True.

        Returns:
            Tuple[np.ndarray, EmbeddingMeta]:
                - np.ndarray: Array of computed embeddings with shape (n_items, embedding_dim).
                - EmbeddingMeta: Metadata about the embeddings (model name, number of items, dimension, creation time, dataset hash).
        """
        dataset_hash = sha1_of_list_str(texts)
        arr_path, meta_path = self._paths(model_name, dataset_hash)

        # Try to load from cache
        try:
            arr, meta = self._load(arr_path, meta_path)
            return arr, meta
        except Exception:
            pass

        # Compute embeddings using OpenAI API
        all_embeddings = []

        # Process in batches to respect API limits and show progress
        if show_progress_bar:
            from tqdm.auto import tqdm

            progress_bar = tqdm(total=len(texts), desc=f"Computing OpenAI embeddings ({model_name})")

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]

            try:
                # Call OpenAI embeddings API
                response = self.client.embeddings.create(input=batch_texts, model=model_name)

                # Extract embeddings from response
                batch_embeddings = [item.embedding for item in response.data]
                all_embeddings.extend(batch_embeddings)

                if show_progress_bar:
                    progress_bar.update(len(batch_texts))

            except Exception as e:
                if show_progress_bar:
                    progress_bar.close()
                raise RuntimeError(f"Failed to get embeddings from OpenAI API: {e}")

        if show_progress_bar:
            progress_bar.close()

        # Convert to numpy array
        emb = np.array(all_embeddings, dtype=np.float32)

        # Normalize embeddings if requested
        if normalize_embeddings:
            emb = self.normalize(emb)

        # Create metadata
        meta = EmbeddingMeta(
            model_name=model_name,
            n_items=len(texts),
            dim=int(emb.shape[1]),
            normalized=normalize_embeddings,
            created_at=time.time(),
            dataset_hash=dataset_hash,
        )

        # Save to cache
        self._save(emb, meta, arr_path, meta_path)
        return emb, meta
