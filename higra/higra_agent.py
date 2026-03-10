import json
import time
import os
from typing import Any, Tuple

from higra_agent.logger import logger
from higra_agent.model_cache import ModelCache
from higra_agent.seeker.seeker import Seeker
from higra_agent.librarian.librarian import Librarian
from higra_agent.retriever.higra_retriever import HiGraRetriever, HierarchicalKnowledgeGraph


# Preload embedding and model cache once at startup
ModelCache.preload_all()


class HiGraAgent:
    """
    HiGraAgent: A unified interface for hierarchical retrieval and adaptive reasoning
    in open-domain multi-hop question answering.
    """

    def __init__(
        self,
        higra_path: str,
        higra_embedding_path: str,
        api_key: str,
        model_name: str,
        reasoning_mode: str,
        retrieval_mode: str,
        retrieval_config_dict: dict
    ) -> None:
        """
        Initialize the HiGraAgent instance.

        Args:
            higra_path (str): Path to the serialized hierarchical graph data.
            higra_embedding_path (str): Path to store or load graph embeddings.
            api_key (str): API key for model access.
            model_name (str): Model identifier (e.g., OpenAI model name).
            reasoning_mode (str): Reasoning strategy ("single_step_reasoning" or "adaptive_reasoning").
            retrieval_mode (str): Retrieval strategy ("higra_retriever" or "hybrid").
        """
        self._validate_modes(reasoning_mode, retrieval_mode)
        self.reasoning_mode = reasoning_mode
        self.retrieval_mode = retrieval_mode

        with open(higra_path, "r", encoding="utf-8") as f:
            higra_data = json.load(f)

        higra = HierarchicalKnowledgeGraph.model_validate(higra_data)
        higra.print_graph_report()

        retriever = HiGraRetriever(
            higra=higra,
            embedder=ModelCache._embedding_model,
            embedding_save_path=higra_embedding_path,
            api_key=api_key,
            model_name=model_name,
            config_dict=retrieval_config_dict
        )

        self.librarian = Librarian(
            api_key=api_key,
            model_name=model_name,
            retriever=retriever,
        )

        self.seeker = Seeker(
            retriever=retriever,
            api_key=api_key,
            model_name=model_name,
            retrieval_mode=retrieval_mode,
        )

    @staticmethod
    def _validate_modes(reasoning_mode: str, retrieval_mode: str) -> None:
        """
        Validate reasoning and retrieval mode arguments.

        Raises:
            ValueError: If any of the provided modes are invalid.
        """
        allowed_reasoning = {"single_step_reasoning", "adaptive_reasoning"}
        allowed_retrieval = {"higra_retriever", "hybrid"}

        if reasoning_mode not in allowed_reasoning:
            raise ValueError(
                f"Invalid reasoning mode '{reasoning_mode}'. "
                f"Allowed values: {sorted(allowed_reasoning)}"
            )

        if retrieval_mode not in allowed_retrieval:
            raise ValueError(
                f"Invalid retrieval mode '{retrieval_mode}'. "
                f"Allowed values: {sorted(allowed_retrieval)}"
            )

    async def run(self, question: str) -> Tuple[str, Any, float, Any]:
        """
        Execute the HiGraAgent pipeline for a given question.

        Args:
            question (str): The input question.

        Returns:
            Tuple[str, Any, float, Any]: 
                - Prediction text
                - Usage statistics
                - Execution time
                - Reasoning history or context
        """
        if self.reasoning_mode == "adaptive_reasoning":
            prediction, usage, timing, history = await self.seeker.run(question)
            pred = self._format_prediction(prediction)
            return pred, usage, timing, history

        # Single-step reasoning with librarian
        start_time = time.time()
        response, usage, context = await self.librarian.run_multiple(
            [question],
            retrieval_mode=self.retrieval_mode,
        )
        timing = time.time() - start_time

        pred = self._format_prediction(response)
        return pred, usage, timing, context

    @staticmethod
    def _format_prediction(prediction: Any) -> str:
        """
        Standardize prediction output into a readable string.

        Args:
            prediction (Any): Prediction result, can be dict, list, or string.

        Returns:
            str: Human-readable prediction.
        """
        if isinstance(prediction, dict):
            answer = prediction.get("answer", "")
            evidence = prediction.get("evidence", "")
            return f"{answer}. {evidence}"
        elif isinstance(prediction, list) and prediction:
            answer = prediction[0].get("answer", "")
            evidence = prediction[0].get("evidence", "")
            return f"{answer}. {evidence}"
        return str(prediction)