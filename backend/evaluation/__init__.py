"""
Evaluation module for ChatMyCV.

Provides:
- Custom LLM-as-Judge evaluation (faithfulness, relevance, citation)
- Simple retrieval metrics (hit rate, MRR, P@K)
- RAGAS integration (optional)
"""

from backend.evaluation.evaluator import Evaluator, EvaluationResult, EvaluationReport
from backend.evaluation.ragas_evaluator import SimpleRetrievalMetrics

__all__ = [
    "Evaluator",
    "EvaluationResult",
    "EvaluationReport",
    "SimpleRetrievalMetrics",
]
