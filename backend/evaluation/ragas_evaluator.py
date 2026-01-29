#!/usr/bin/env python
# -*- coding:utf-8 -*-

"""
RAGAS Integration for ChatMyCV

Provides retrieval-focused evaluation metrics using the RAGAS framework:
- Context Precision: Are retrieved docs relevant to the question?
- Context Recall: Did we retrieve all necessary information?
- Faithfulness: Is the answer faithful to the context?
- Answer Relevancy: Is the answer relevant to the question?

Install: pip install ragas
"""

import sys
sys.path.append("./")
sys.path.append("../")

import asyncio
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from utils.app_logger import LoggerSetup

logger = LoggerSetup("RagasEvaluator").logger

# Check if RAGAS is installed
try:
    from ragas import evaluate
    from ragas.metrics import (
        context_precision,
        context_recall,
        faithfulness,
        answer_relevancy,
    )
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from datasets import Dataset
    RAGAS_AVAILABLE = True
except ImportError:
    RAGAS_AVAILABLE = False
    logger.warning("RAGAS not installed. Run: pip install ragas datasets")


@dataclass
class RagasResult:
    """RAGAS evaluation result for a single test case."""
    test_id: str
    question: str
    context_precision: float
    context_recall: float
    faithfulness: float
    answer_relevancy: float

    @property
    def overall_score(self) -> float:
        return (
            self.context_precision * 0.25 +
            self.context_recall * 0.25 +
            self.faithfulness * 0.25 +
            self.answer_relevancy * 0.25
        )


class RagasEvaluator:
    """
    RAGAS-based evaluator for RAG retrieval quality.

    Requires:
    - pip install ragas datasets langchain-openai
    """

    def __init__(self, azure_llm=None, azure_embeddings=None):
        if not RAGAS_AVAILABLE:
            raise ImportError(
                "RAGAS is not installed. Please run: pip install ragas datasets"
            )

        self.llm = azure_llm
        self.embeddings = azure_embeddings

    def _prepare_dataset(
        self,
        questions: List[str],
        answers: List[str],
        contexts: List[List[str]],
        ground_truths: List[str] = None
    ) -> "Dataset":
        """Prepare data in RAGAS expected format."""
        data = {
            "question": questions,
            "answer": answers,
            "contexts": contexts,
        }
        if ground_truths:
            data["ground_truth"] = ground_truths

        return Dataset.from_dict(data)

    def evaluate_batch(
        self,
        questions: List[str],
        answers: List[str],
        contexts: List[List[str]],
        ground_truths: List[str] = None,
        metrics: List[str] = None
    ) -> Dict[str, Any]:
        """
        Evaluate a batch of Q&A pairs using RAGAS.

        Args:
            questions: List of questions
            answers: List of generated answers
            contexts: List of retrieved context lists (each item is a list of retrieved docs)
            ground_truths: Optional list of expected answers (for context_recall)
            metrics: Which metrics to compute. Default: all available

        Returns:
            Dictionary with metric scores
        """
        dataset = self._prepare_dataset(questions, answers, contexts, ground_truths)

        # Select metrics
        available_metrics = []
        if metrics is None or "context_precision" in metrics:
            available_metrics.append(context_precision)
        if metrics is None or "faithfulness" in metrics:
            available_metrics.append(faithfulness)
        if metrics is None or "answer_relevancy" in metrics:
            available_metrics.append(answer_relevancy)
        if ground_truths and (metrics is None or "context_recall" in metrics):
            available_metrics.append(context_recall)

        # Run evaluation
        result = evaluate(
            dataset,
            metrics=available_metrics,
            llm=self.llm,
            embeddings=self.embeddings,
        )

        return result.to_pandas().to_dict()


def create_ragas_evaluator_with_azure():
    """
    Create a RAGAS evaluator configured with Azure OpenAI.

    Requires:
    - pip install langchain-openai
    """
    try:
        from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
        import os

        llm = AzureChatOpenAI(
            azure_deployment=os.getenv("AZURE_OPENAI_LLM_MODEL"),
            api_version="2024-02-15-preview",
        )
        embeddings = AzureOpenAIEmbeddings(
            azure_deployment=os.getenv("AZURE_OPENAI_EMBEDDING_MODEL"),
            api_version="2024-02-15-preview",
        )

        wrapped_llm = LangchainLLMWrapper(llm)
        wrapped_embeddings = LangchainEmbeddingsWrapper(embeddings)

        return RagasEvaluator(
            azure_llm=wrapped_llm,
            azure_embeddings=wrapped_embeddings
        )
    except ImportError:
        logger.error("langchain-openai not installed. Run: pip install langchain-openai")
        raise


# Alternative: Simple retrieval metrics without RAGAS
class SimpleRetrievalMetrics:
    """
    Simple retrieval quality metrics without RAGAS dependency.
    """

    @staticmethod
    def hit_rate(
        retrieved_sources: List[str],
        expected_sources: List[str]
    ) -> float:
        """
        Calculate hit rate: what fraction of expected sources were retrieved?

        Args:
            retrieved_sources: List of retrieved document source names/ids
            expected_sources: List of expected source names/ids

        Returns:
            Hit rate between 0.0 and 1.0
        """
        if not expected_sources:
            return 1.0  # No expected sources means any retrieval is fine

        hits = sum(1 for exp in expected_sources if any(exp.lower() in ret.lower() for ret in retrieved_sources))
        return hits / len(expected_sources)

    @staticmethod
    def mrr(
        retrieved_sources: List[str],
        expected_sources: List[str]
    ) -> float:
        """
        Calculate Mean Reciprocal Rank.

        Args:
            retrieved_sources: Ordered list of retrieved sources
            expected_sources: List of relevant source names

        Returns:
            MRR score between 0.0 and 1.0
        """
        if not expected_sources:
            return 1.0

        for i, retrieved in enumerate(retrieved_sources, 1):
            for expected in expected_sources:
                if expected.lower() in retrieved.lower():
                    return 1.0 / i
        return 0.0

    @staticmethod
    def precision_at_k(
        retrieved_sources: List[str],
        expected_sources: List[str],
        k: int = 5
    ) -> float:
        """
        Calculate Precision@K.

        Args:
            retrieved_sources: List of retrieved sources
            expected_sources: List of relevant sources
            k: Number of top results to consider

        Returns:
            Precision@K score
        """
        if not retrieved_sources or not expected_sources:
            return 0.0 if expected_sources else 1.0

        top_k = retrieved_sources[:k]
        relevant_in_k = sum(
            1 for ret in top_k
            if any(exp.lower() in ret.lower() for exp in expected_sources)
        )
        return relevant_in_k / k


if __name__ == "__main__":
    # Test simple metrics
    metrics = SimpleRetrievalMetrics()

    retrieved = ["Work Experience - Software Engineer", "Skills - Python", "Education - BS"]
    expected = ["Work Experience", "Skills"]

    print(f"Hit Rate: {metrics.hit_rate(retrieved, expected)}")
    print(f"MRR: {metrics.mrr(retrieved, expected)}")
    print(f"P@3: {metrics.precision_at_k(retrieved, expected, k=3)}")
