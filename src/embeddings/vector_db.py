"""
ChromaDB-based vector database for storing and retrieving paper embeddings.
"""

from __future__ import annotations

import os
import random
import re
from typing import Dict, List, Optional, Tuple

import chromadb
import numpy as np
import pandas as pd
from chromadb.config import Settings

from utils.utils import join_title_abstract

from .embeddings import EmbeddingCache, OpenAIEmbeddingCache, SPECTER2EmbeddingCache

RANDOM_STATE = int(os.getenv("RANDOM_STATE"))


class ChromaVectorDB:
    """
    ChromaDB-based vector database for storing and retrieving paper embeddings.

    This class provides functionality to:
    - Store papers with their embeddings in ChromaDB
    - Perform similarity search to find similar papers
    - Use pre-computed embeddings from the embedding cache
    """

    def __init__(
        self,
        collection_name: str,
        embedding_model: str,
        db_path: str = "./vector_db/chroma",
        embedding_cache_dir: str = "./.embeddings_cache",
    ):
        """
        Initialize the ChromaDB vector database.

        Args:
            db_path: Path where ChromaDB will store its data
            collection_name: Name of the ChromaDB collection
            embedding_model: Model to use for embeddings (should match cached embeddings)
            embedding_cache_dir: Directory where embeddings are cached
        """
        self.db_path = db_path
        self.collection_name = collection_name
        self.embedding_model = embedding_model
        self.embedding_cache_dir = embedding_cache_dir

        # create database directory
        os.makedirs(db_path, exist_ok=True)

        assert os.path.isdir(embedding_cache_dir), f"Embedding cache directory does not exist: {embedding_cache_dir} (current dir: {os.getcwd()})"

        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(path=db_path, settings=Settings(anonymized_telemetry=False))

        # Initialize embedding cache
        if EmbeddingCache.is_specter2_model(embedding_model):
            self.embedding_cache = SPECTER2EmbeddingCache(cache_dir=embedding_cache_dir)
        elif EmbeddingCache.is_openai_model(embedding_model):
            self.embedding_cache = OpenAIEmbeddingCache(cache_dir=embedding_cache_dir)
        else:
            self.embedding_cache = EmbeddingCache(cache_dir=embedding_cache_dir)

        # Collection will be created when data is added
        self._collection = None

        # set random seed for reproducibility
        random.seed(RANDOM_STATE)

    def _get_collection(self) -> Optional[chromadb.api.models.Collection.Collection]:
        """Get or create the ChromaDB collection."""
        if self._collection is None:
            try:
                self._collection = self.client.get_collection(name=self.collection_name)

            # catch any exception when collection doesn't exist
            except Exception as e:
                print(e)
                return None
        return self._collection

    def add_papers(
        self,
        df: pd.DataFrame,
        id_col: str = "id",
        title_col: str = "title",
        abstract_col: str = "abstract",
        label_col: str = "label",
        batch_size: int = 100,
        normalize_embeddings: bool = True,
    ) -> None:
        """
        Add papers to the vector database.

        Args:
            df: DataFrame containing papers
            id_col: Column name for paper IDs
            title_col: Column name for paper titles
            abstract_col: Column name for paper abstracts
            label_col: Column name for paper labels
            batch_size: Batch size for adding to ChromaDB
            normalize_embeddings: Whether to normalize embeddings
        """
        print(f"Adding {len(df)} papers to vector database...")

        # Prepare texts (title + abstract)
        texts = join_title_abstract(df)

        # Get embeddings from cache
        embeddings, meta = self.embedding_cache.compute(
            texts,
            self.embedding_model,
            batch_size=128,
            normalize_embeddings=False,  # We'll normalize separately if needed
            device=None,
        )

        # Normalize if requested
        if normalize_embeddings:
            embeddings = self.embedding_cache.normalize(embeddings)

        print(f"Using embeddings: shape={embeddings.shape}, model={meta.model_name}")

        # Create or get collection
        if self._collection is None:
            self._collection = self.client.create_collection(name=self.collection_name, metadata={"hnsw:space": "cosine"})  # Use cosine similarity

        # Prepare data for ChromaDB
        ids = [str(paper_id) for paper_id in df[id_col].tolist()]
        documents = texts
        metadatas = []

        for _, row in df.iterrows():
            metadata = {
                "title": str(row[title_col]),
                "abstract": str(row[abstract_col]),
                "label": bool(row[label_col]),
                "text": join_title_abstract(row.to_frame().T)[0],
            }
            metadatas.append(metadata)

        # Add to ChromaDB in batches
        total_added = 0
        for i in range(0, len(ids), batch_size):
            batch_ids = ids[i : i + batch_size]
            batch_documents = documents[i : i + batch_size]
            batch_embeddings = embeddings[i : i + batch_size].tolist()
            batch_metadatas = metadatas[i : i + batch_size]

            self._collection.add(ids=batch_ids, documents=batch_documents, embeddings=batch_embeddings, metadatas=batch_metadatas)
            total_added += len(batch_ids)
            print(f"Added batch: {total_added}/{len(ids)} papers")

        print(f"Successfully added {total_added} papers to vector database")

    def search_similar(
        self,
        query_text: str,
        n_results: int = 3,
        label: Optional[str] = None,
        include_metadata: bool = True,
        include_self: bool = False,
    ) -> Dict:
        """
        Search for similar papers based on query text.

        Args:
            query_text: Text to search for (will be embedded)
            n_results: Number of similar papers to return
            label: Optional label to filter results by (either INCLUDE, EXCLUDE, or None)
            include_metadata: Whether to include paper metadata in results
            include_self: Whether to include the query paper if it exists in the database (based on text match)

        Returns:
            Dictionary with search results from ChromaDB
        """
        assert label in (None, "INCLUDE", "EXCLUDE"), "Label must be either 'INCLUDE', 'EXCLUDE', or None"

        collection = self._get_collection()
        if collection is None:
            raise ValueError("No collection found. Please add papers first.")

        # Embed the query text
        query_embeddings, _ = self.embedding_cache.compute(
            [query_text],
            self.embedding_model,
            batch_size=1,
            normalize_embeddings=True,  # Match normalization used in storage
            device=None,
        )

        if label is not None:
            where_clause = {"label": True if label == "INCLUDE" else False}
        else:
            where_clause = None

        # Search in ChromaDB
        results = collection.query(
            query_embeddings=query_embeddings.tolist(),
            n_results=n_results,
            include=["metadatas", "documents", "distances"] if include_metadata else ["documents", "distances"],
            where=where_clause,
            where_document={"$not_contains": query_text} if not include_self else None,
        )

        return results

    def search_random(
        self,
        n_results: int = 3,
        label: Optional[str] = None,
    ):
        """
        Retrieve random papers from the database.

        Args:
            n_results: Number of random papers to return
            label: Optional label to filter results by (either INCLUDE, EXCLUDE, or None)

        Returns:
            Dictionary with random results from ChromaDB
        """
        collection = self._get_collection()
        if collection is None:
            raise ValueError("No collection found. Please add papers first.")

        # get all ids to then choose random ones
        all_ids = collection.get(include=["metadatas"])["ids"]

        # filter by label if specified
        if label is not None:
            filtered_ids = []
            metadatas = collection.get(ids=all_ids, include=["metadatas"])["metadatas"]
            for paper_id, metadata in zip(all_ids, metadatas):
                if (label == "INCLUDE" and metadata["label"] is True) or (label == "EXCLUDE" and metadata["label"] is False):
                    filtered_ids.append(paper_id)
            all_ids = filtered_ids

        # sample n_results random ids
        random_ids = random.sample(all_ids, n_results)

        # get and return results
        results = collection.get(ids=random_ids, include=["metadatas", "documents"])
        # nest list 
        results = {key: [value] for key, value in results.items()}
        return results

    def search_similar_by_id(
        self,
        paper_id: str,
        n_results: int = 3,
        include_self: bool = False,
    ) -> Dict:
        """
        Search for papers similar to a given paper ID.

        Args:
            paper_id: ID of the paper to find similar papers for
            n_results: Number of similar papers to return
            include_self: Whether to include the query paper in results

        Returns:
            Dictionary with search results from ChromaDB
        """
        collection = self._get_collection()
        if collection is None:
            raise ValueError("No collection found. Please add papers first.")

        # Get the paper by ID
        paper_result = collection.get(ids=[str(paper_id)], include=["embeddings"])
        if not paper_result["ids"]:
            raise ValueError(f"Paper with ID {paper_id} not found in database")

        # Use the paper's embedding as query
        query_embedding = paper_result["embeddings"][0]

        # Search for similar papers
        results = collection.query(
            query_embeddings=[query_embedding], n_results=n_results + (1 if not include_self else 0), include=["metadatas", "documents", "distances"]
        )

        # Remove self from results if not requested
        if not include_self:
            filtered_results = {"ids": [], "documents": [], "metadatas": [], "distances": []}
            for i, result_id in enumerate(results["ids"][0]):
                if result_id != str(paper_id):
                    filtered_results["ids"].append(result_id)
                    filtered_results["documents"].append(results["documents"][0][i])
                    filtered_results["metadatas"].append(results["metadatas"][0][i])
                    filtered_results["distances"].append(results["distances"][0][i])

            # Take only n_results
            for key in filtered_results:
                filtered_results[key] = filtered_results[key][:n_results]

            results = {
                "ids": [filtered_results["ids"]],
                "documents": [filtered_results["documents"]],
                "metadatas": [filtered_results["metadatas"]],
                "distances": [filtered_results["distances"]],
            }

        return results

    def get_paper_by_id(self, paper_id: str) -> Dict:
        """
        Get a paper by its ID.

        Args:
            paper_id: ID of the paper to retrieve

        Returns:
            Dictionary with paper data
        """
        collection = self._get_collection()
        if collection is None:
            raise ValueError("No collection found. Please add papers first.")

        result = collection.get(ids=[str(paper_id)], include=["metadatas", "documents"])
        if not result["ids"]:
            raise ValueError(f"Paper with ID {paper_id} not found in database")

        return {"id": result["ids"][0], "document": result["documents"][0], "metadata": result["metadatas"][0]}

    def get_collection_info(self) -> Dict:
        """
        Get information about the collection.

        Returns:
            Dictionary with collection statistics
        """
        collection = self._get_collection()
        if collection is None:
            return {"count": 0, "exists": False}

        count = collection.count()
        return {"count": count, "exists": True, "name": self.collection_name, "embedding_model": self.embedding_model}

    def clear_collection(self) -> None:
        """Clear all data from the collection."""
        try:
            self.client.delete_collection(name=self.collection_name)
            self._collection = None
            print(f"Cleared collection: {self.collection_name}")
        except ValueError:
            print(f"Collection {self.collection_name} does not exist")
