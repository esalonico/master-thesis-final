"""
LLM-based classifier for Title & Abstract (TIAB) screening using OpenAI's GPT models and Google's Gemini models.
"""

from __future__ import annotations

import json
import os
import time
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import openai
import pandas as pd
from dotenv import load_dotenv
from google import genai
from tqdm.auto import tqdm

# Load environment variables
load_dotenv()


@dataclass
class LLMClassificationResult:
    """Result of LLM classification for a single paper."""

    id: int  # Paper identifier
    decision: str  # "YES", "NO", or "UNCERTAIN"
    raw_response: str  # Raw LLM response
    processing_time: float  # Time taken for classification
    model_base: str
    model_full: str
    input_tokens: int
    output_tokens: int


class LLMClassifier(ABC):
    """
    Abstract base class for LLM-based classifier for systematic review screening.

    Uses LLM models to classify papers as include/exclude/uncertain
    based on title and abstract content.
    """

    def __init__(
        self,
        model: str,
        use_few_shot: bool = False,
        n_shots: int = 1,
    ):
        """
        Initialize the LLM classifier.

        Args:
            model: Model identifier to use
            use_few_shot: Whether to use few-shot examples (placeholder for future)
            n_shots: Number of few-shot examples to use
        """
        self.model = model
        self.use_few_shot = use_few_shot
        self.n_shots = n_shots

        # default inclusion/exclusion criteria - can be customized
        self.inclusion_criteria = ""
        self.exclusion_criteria = ""

        # costs
        self._COST_PER_1M_TOKENS_INPUT = None
        self._COST_PER_1M_TOKENS_OUTPUT = None
        self._get_model_costs()

        # initialize the specific LLM client
        self._init_client()

        # initialize vector database if few-shot is used
        if self.use_few_shot:
            self._init_vector_db()

    @abstractmethod
    def _init_client(self) -> None:
        """Initialize the specific LLM client (OpenAI, Gemini, etc.)."""
        pass

    def _init_vector_db(self) -> None:
        """Initialize the vector database for few-shot prompting."""
        from embeddings.vector_db import ChromaVectorDB

        CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH")
        CHROMA_COLLECTION_NAME = os.getenv("CHROMA_COLLECTION_NAME")
        CHROMA_EMBEDDING_MODEL = os.getenv("CHROMA_EMBEDDING_MODEL")
        CACHE_DIR_EMBEDDINGS = os.getenv("CACHE_DIR_EMBEDDINGS")

        assert CHROMA_DB_PATH is not None, "CHROMA_DB_PATH environment variable not set."
        assert CHROMA_COLLECTION_NAME is not None, "CHROMA_COLLECTION_NAME environment variable not set."
        assert CHROMA_EMBEDDING_MODEL is not None, "CHROMA_EMBEDDING_MODEL environment variable not set."
        assert CACHE_DIR_EMBEDDINGS is not None, "CACHE_DIR_EMBEDDINGS environment variable not set."

        # initialize Chroma vector database
        self.vector_db = ChromaVectorDB(
            db_path=CHROMA_DB_PATH,
            collection_name=CHROMA_COLLECTION_NAME,
            embedding_model=CHROMA_EMBEDDING_MODEL,
            embedding_cache_dir=CACHE_DIR_EMBEDDINGS,
        )

    @abstractmethod
    def _call_llm(self, prompt: str) -> Tuple[str, int, int, str]:
        """
        Call the specific LLM API.

        Args:
            prompt: The input prompt

        Returns:
            Tuple of (response_text, input_tokens, output_tokens, full_model_name)
        """
        pass

    def set_criteria(self, inclusion_criteria: str, exclusion_criteria: str) -> None:
        """
        Set the inclusion and exclusion criteria for screening.

        Args:
            inclusion_criteria: Description of what papers to include
            exclusion_criteria: Description of what papers to exclude
        """
        self.inclusion_criteria = inclusion_criteria
        self.exclusion_criteria = exclusion_criteria

    def _get_model_costs(self, costs_file_json: str = "src/llm/models_costs.json") -> None:
        """Read model-specific cost parameters from JSON file and set attributes."""
        # Construct absolute path to costs file relative to the script location
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.join(script_dir, "..", "..")
        costs_file_path = os.path.join(project_root, costs_file_json)
        costs_file_path = os.path.normpath(costs_file_path)

        with open(costs_file_path, "r") as f:
            costs = json.load(f)

        model_costs = costs.get(self.model)

        if not model_costs:
            raise ValueError(f"Costs not defined for model {self.model} in file {costs_file_json}")

        self._COST_PER_1M_TOKENS_INPUT = model_costs.get("COST_PER_1M_TOKENS_INPUT")
        self._COST_PER_1M_TOKENS_OUTPUT = model_costs.get("COST_PER_1M_TOKENS_OUTPUT")

    def _query_few_shot_examples(self, title: str, abstract: str) -> str:
        """
        Query the vector database for few-shot examples.
        n_shots: number of examples to retrieve for each label (INCLUDE, EXCLUDE, UNCERTAIN)
        """
        search_query = f"{title} {abstract}"

        results = []

        # include
        if self.shot_selection_strategy == "similarity":
            include_results = self.vector_db.search_similar(
                query_text=search_query,
                n_results=self.n_shots,
                label="INCLUDE",
                include_metadata=True,
                include_self=False,
            )
        else:  # random selection
            include_results = self.vector_db.search_random(
                n_results=self.n_shots,
                label="INCLUDE",
            )
        results.extend(include_results["metadatas"])

        # exclude
        if self.shot_selection_strategy == "similarity":
            exclude_results = self.vector_db.search_similar(
                query_text=search_query,
                n_results=self.n_shots,
                label="EXCLUDE",
                include_metadata=True,
                include_self=False,
            )
        else:  # random selection
            exclude_results = self.vector_db.search_random(
                n_results=self.n_shots,
                label="EXCLUDE",
            )
        results.extend(exclude_results["metadatas"])

        # flatten list
        results = [item for sublist in results for item in sublist]

        # output
        output = "EXAMPLES:\n\n"
        for res in results:
            label_mapped = "YES" if res["label"] is True else "NO"
            output += f"Title: {res['title']}\n"
            output += f"Abstract: {res['abstract']}\n"
            output += f"Label: {label_mapped}\n\n"

        return output

    def _create_prompt(self, title: str, abstract: str) -> str:
        """
        Create the prompt for the LLM based on title, abstract, and criteria.

        Args:
            title: Paper title
            abstract: Paper abstract

        Returns:
            Formatted prompt string
        """
        if self.use_few_shot:
            few_shot_examples = self._query_few_shot_examples(title, abstract)
        else:
            few_shot_examples = ""

        prompt = f"""You are an expert researcher conducting a systematic review.
            Your task is to screen papers for inclusion/exclusion based on their title and abstract.

            INCLUSION CRITERIA:
            {self.inclusion_criteria}

            EXCLUSION CRITERIA:
            {self.exclusion_criteria}

            {few_shot_examples}

            PAPER TO EVALUATE:
            Title: {title}

            Abstract: {abstract}

            INSTRUCTIONS:
            1. Carefully read the title and abstract
            2. Evaluate whether this paper meets the inclusion criteria
            3. Check if it meets any exclusion criteria
            4. Make a decision: YES (include), NO (exclude), or UNCERTAIN (unclear/borderline)

            Respond with only: YES, NO, or UNCERTAIN"""

        return prompt

    def classify_single(self, id: int, title: str, abstract: str) -> LLMClassificationResult:
        """
        Classify a single paper based on title and abstract.

        Args:
            id: Unique identifier for the paper
            title: Paper title
            abstract: Paper abstract

        Returns:
            LLMClassificationResult with decision and metadata
        """
        start_time = time.time()

        try:
            prompt = self._create_prompt(title, abstract)

            raw_response, input_tokens, output_tokens, full_model_name = self._call_llm(prompt)

            processing_time = round(time.time() - start_time, 3)

            # Parse simple response (just decision)
            decision = raw_response.strip().upper()

            # Validate decision
            if decision not in ["YES", "NO", "UNCERTAIN"]:
                # Try to extract decision from response
                if "YES" in decision:
                    decision = "YES"
                elif "NO" in decision:
                    decision = "NO"
                else:
                    decision = "UNCERTAIN"

            return LLMClassificationResult(
                id=int(id),
                decision=decision,
                raw_response=raw_response,
                processing_time=processing_time,
                model_base=self.model,
                model_full=full_model_name,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
            )

        except Exception as e:
            raise e

    def classify_papers(self, papers: List[Tuple[int, str, str]], parallel: bool = False, n_workers: int = 5) -> List[LLMClassificationResult]:
        """
        Classify multiple papers in a loop or in parallel.

        Args:
            papers: List of (id, title, abstract) tuples
            parallel: Whether to use parallel processing
            n_workers: Number of worker processes for parallel execution
        Returns:
            List of LLMClassificationResult objects
        """
        # sequential processing
        if not parallel:
            results = []
            for i, (id, title, abstract) in enumerate(tqdm(papers, desc=f"Classifying papers ({self.model})")):
                result = self.classify_single(id, title, abstract)
                results.append(result)
            return results

        # parallel processing
        else:
            results = []

            with ThreadPoolExecutor(max_workers=n_workers) as executor:
                # Submit all tasks
                future_to_paper = {
                    executor.submit(self.classify_single, id, title, abstract): (id, title, abstract) for id, title, abstract in papers
                }

                # Process completed tasks with progress bar
                for future in tqdm(
                    as_completed(future_to_paper), total=len(papers), desc=f"Classifying papers ({self.model}, {n_workers} parallel workers)"
                ):
                    try:
                        result = future.result()
                        results.append(result)
                    except Exception as e:
                        id, title, abstract = future_to_paper[future]
                        print(f"Error processing paper {id}: {e}")

            return results

    def classify_dataframe(
        self,
        df: pd.DataFrame,
        title_col: str = "title",
        abstract_col: str = "abstract",
        parallel: bool = False,
        n_workers: int = 5,
    ) -> pd.DataFrame:
        """
        Classify papers from a pandas DataFrame.

        Args:
            df: DataFrame containing papers
            title_col: Name of title column
            abstract_col: Name of abstract column

        Returns:
            DataFrame with additional columns for LLM results
        """
        papers = [(row["id"], row[title_col], row[abstract_col]) for _, row in df.iterrows()]
        results: List[LLMClassificationResult] = self.classify_papers(papers, parallel=parallel, n_workers=n_workers)

        # Create result DataFrame
        metadata_df = pd.DataFrame(results)

        results_df = pd.merge(df, metadata_df, on="id")

        return results_df

    def compute_costs(self, results_df: pd.DataFrame) -> dict:
        """
        Compute cost estimates based on token usage in results DataFrame.
        Args:
            results_df: DataFrame with classification results including token counts
        Returns:
            Dictionary with cost estimates
        """
        tot_input_tokens = results_df.input_tokens.sum()
        tot_output_tokens = results_df.output_tokens.sum()
        tot_cost = (tot_input_tokens / 1_000_000) * self._COST_PER_1M_TOKENS_INPUT + (tot_output_tokens / 1_000_000) * self._COST_PER_1M_TOKENS_OUTPUT
        cost_per_paper = tot_cost / len(results_df)
        cost_per_1k_papers = cost_per_paper * 1000

        return {
            "cost_per_paper": round(float(cost_per_paper), 4),
            "cost_per_1k_papers": round(float(cost_per_1k_papers), 3),
            "total_cost": round(float(tot_cost), 3),
            "n_papers": len(results_df),
        }


class OpenAIClassifier(LLMClassifier):
    """OpenAI GPT-based classifier implementation."""

    def __init__(
        self,
        model: str,
        api_key: Optional[str] = None,
        use_few_shot: bool = False,
        n_shots: int = 1,
        shot_selection_strategy: str = "similarity",  # "similarity" or "random"
    ):
        """
        Initialize the OpenAI classifier.

        Args:
            model: OpenAI model to use (e.g., "gpt-4o-mini", "gpt-4o")
            api_key: OpenAI API key (if None, reads from environment)
            use_few_shot: Whether to use few-shot examples
            n_shots: Number of few-shot examples to use
            shot_selection_strategy: Strategy for selecting few-shot examples ("similarity" or "random")
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key not found. Please set OPENAI_API_KEY environment variable.")

        self.shot_selection_strategy = shot_selection_strategy

        super().__init__(model, use_few_shot, n_shots)

    def _init_client(self) -> None:
        """Initialize the OpenAI client."""
        self.client = openai.OpenAI(api_key=self.api_key)

    def _call_llm(self, prompt: str) -> Tuple[str, int, int, str]:
        """
        Call OpenAI API.

        Args:
            prompt: The input prompt

        Returns:
            Tuple of (response_text, input_tokens, output_tokens, full_model_name)
        """
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )

        raw_response = response.choices[0].message.content
        input_tokens = response.usage.prompt_tokens
        output_tokens = response.usage.completion_tokens
        full_model_name = response.model

        return raw_response, input_tokens, output_tokens, full_model_name


class GeminiClassifier(LLMClassifier):
    """Google Gemini-based classifier implementation."""

    def __init__(
        self,
        model: str,
        api_key: Optional[str] = None,
        use_few_shot: bool = False,
        n_shots: int = 1,
        shot_selection_strategy: str = "similarity",  # "similarity" or "random"
    ):
        """
        Initialize the Gemini classifier.

        Args:
            model: Gemini model to use (e.g., "gemini-1.5-flash", "gemini-1.5-pro")
            api_key: Google API key (if None, reads from environment)
            use_few_shot: Whether to use few-shot examples
            n_shots: Number of few-shot examples to use
            shot_selection_strategy: Strategy for selecting few-shot examples ("similarity" or "random")
        """
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("Google API key not found. Please set GEMINI_API_KEY environment variable.")

        self.shot_selection_strategy = shot_selection_strategy

        super().__init__(model, use_few_shot, n_shots)

    def _init_client(self) -> None:
        """Initialize the Gemini client."""
        self.client = genai.Client()

    def _call_llm(self, prompt: str) -> Tuple[str, int, int, str]:
        """
        Call Gemini API.

        Args:
            prompt: The input prompt

        Returns:
            Tuple of (response_text, input_tokens, output_tokens, full_model_name)
        """
        response = self.client.models.generate_content(model=self.model, contents=prompt)
        raw_response = response.text

        try:
            input_tokens = response.usage_metadata.prompt_token_count
            output_tokens = response.usage_metadata.candidates_token_count  # thinking tokens are inclued here
        except AttributeError:
            input_tokens = None
            output_tokens = None

        full_model_name = response.model_version

        return raw_response, input_tokens, output_tokens, full_model_name


class MedGemmaClassifier(LLMClassifier):
    """MedGemma-based classifier implementation using vLLM endpoint."""

    def __init__(
        self,
        model: str = "google/medgemma-27b-text-it",
        api_key: Optional[str] = None,
        base_url: str = "http://vllm.mevidence.ai/v1",
        use_few_shot: bool = False,
        n_shots: int = 1,
        shot_selection_strategy: str = "similarity",  # "similarity" or "random"
    ):
        """
        Initialize the MedGemma classifier.

        Args:
            model: MedGemma model to use (e.g., "google/medgemma-27b-text-it")
            api_key: vLLM API key (if None, reads from VLLM_API_KEY environment)
            base_url: vLLM base URL endpoint
            use_few_shot: Whether to use few-shot examples
            n_shots: Number of few-shot examples to use
            shot_selection_strategy: Strategy for selecting few-shot examples ("similarity" or "random")
        """
        self.api_key = api_key or os.getenv("VLLM_API_KEY")
        if not self.api_key:
            raise ValueError("vLLM API key not found. Please set VLLM_API_KEY environment variable.")

        self.base_url = base_url
        self.shot_selection_strategy = shot_selection_strategy

        super().__init__(model, use_few_shot, n_shots)

    def _init_client(self) -> None:
        """Initialize the OpenAI client pointing to vLLM endpoint."""
        self.client = openai.OpenAI(api_key=self.api_key, base_url=self.base_url)

    def _call_llm(self, prompt: str) -> Tuple[str, int, int, str]:
        """
        Call MedGemma via vLLM endpoint using OpenAI wrapper.

        Args:
            prompt: The input prompt

        Returns:
            Tuple of (response_text, input_tokens, output_tokens, full_model_name)
        """
        # Construct messages with system prompt to avoid reasoning tokens
        messages = [
            {
                "role": "system",
                "content": "You are a medical assistant helping with systematic review screening. Do NOT output reasoning tokens, only the final answer.",
            },
            {"role": "user", "content": prompt},
        ]

        response = self.client.chat.completions.create(model=self.model, max_tokens=4096, messages=messages, temperature=0)

        raw_response = response.choices[0].message.content.strip()
        input_tokens = response.usage.prompt_tokens
        output_tokens = response.usage.completion_tokens
        full_model_name = response.model

        return raw_response, input_tokens, output_tokens, full_model_name


def evaluate_llm_classifier(y_true: pd.Series, llm_decisions: pd.Series, uncertain_as_positive: bool = True) -> Dict[str, float]:
    """
    Evaluate LLM classifier performance against ground truth labels.

    Args:
        y_true: True labels (True/False for include/exclude)
        llm_decisions: LLM decisions ("YES"/"NO"/"UNCERTAIN")
        uncertain_as_positive: Whether to treat UNCERTAIN as positive (include)

    Returns:
        Dictionary with evaluation metrics
    """
    from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score

    # Convert LLM decisions to binary
    if uncertain_as_positive:
        y_pred = llm_decisions.apply(lambda x: True if x in ["YES", "UNCERTAIN"] else False)
    else:
        y_pred = llm_decisions.apply(lambda x: True if x == "YES" else False)

    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    # Calculate specificity (True Negative Rate)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    # Additional metrics
    total_uncertain = (llm_decisions == "UNCERTAIN").sum()
    uncertain_rate = total_uncertain / len(llm_decisions)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "specificity": specificity,
        "uncertain_rate": uncertain_rate,
        "total_uncertain": total_uncertain,
        "total_samples": len(llm_decisions),
    }
