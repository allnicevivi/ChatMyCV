#!/usr/bin/env python
# -*- coding:utf-8 -*-

"""
Evaluation Runner for ChatMyCV

Runs comprehensive evaluation using:
1. Custom LLM-as-Judge metrics (faithfulness, relevance, citation)
2. Simple retrieval metrics (hit rate, MRR, P@K)
3. RAGAS metrics (optional, if installed)

Usage:
    python backend/evaluation/run_evaluation.py --language en
    python backend/evaluation/run_evaluation.py --language zhtw --use-ragas
"""

import sys
sys.path.append("./")
sys.path.append("../")

import json
import asyncio
import argparse
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any

from backend.services.chat_serv import ChatService
from backend.evaluation.evaluator import Evaluator, EvaluationReport, EvaluationResult
from backend.evaluation.ragas_evaluator import SimpleRetrievalMetrics
from utils.app_logger import LoggerSetup

logger = LoggerSetup("EvalRunner").logger

EVAL_DIR = Path(__file__).parent
OUTPUT_DIR = EVAL_DIR / "results"


async def run_single_test(
    chat_service: ChatService,
    evaluator: Evaluator,
    test_case: Dict[str, Any],
    language: str
) -> Dict[str, Any]:
    """Run a single test case through the RAG pipeline and evaluate."""

    question = test_case["question"]
    test_id = test_case["id"]
    expected_sources = test_case.get("expected_sources", [])

    logger.info(f"Running test: {test_id}")

    # Get response from RAG system
    response = await chat_service.achat(
        query=question,
        language=language,
        persona="engineer"  # Use consistent persona for evaluation
    )

    answer = response.get("answer", "")
    retrieved_docs = response.get("retrieved_docs", [])

    # Format context from retrieved docs
    context = "\n\n".join([
        f"[{doc.get('metadata', {}).get('header', 'Unknown')}]\n{doc.get('text', '')}"
        for doc in retrieved_docs
    ])

    # Get source names from retrieved docs
    retrieved_sources = [
        doc.get("metadata", {}).get("header", "Unknown")
        for doc in retrieved_docs
    ]

    # Run LLM-as-Judge evaluation
    eval_result = await evaluator.evaluate(
        test_id=test_id,
        question=question,
        answer=answer,
        context=context,
        expected_sources=expected_sources
    )

    # Calculate retrieval metrics
    retrieval_metrics = SimpleRetrievalMetrics()
    hit_rate = retrieval_metrics.hit_rate(retrieved_sources, expected_sources)
    mrr = retrieval_metrics.mrr(retrieved_sources, expected_sources)
    p_at_k = retrieval_metrics.precision_at_k(retrieved_sources, expected_sources, k=5)

    return {
        "test_id": test_id,
        "question": question,
        "answer": answer,
        "context": context[:1000] + "..." if len(context) > 1000 else context,
        "retrieved_sources": retrieved_sources,
        "expected_sources": expected_sources,
        "eval_result": eval_result,
        "retrieval_metrics": {
            "hit_rate": hit_rate,
            "mrr": mrr,
            "precision_at_5": p_at_k
        }
    }


async def run_evaluation(
    test_cases_path: Path,
    language: str = "en",
    output_path: Path = None
) -> Dict[str, Any]:
    """Run full evaluation suite."""

    # Load test cases
    with open(test_cases_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    test_cases = data.get("test_cases", [])

    logger.info(f"Loaded {len(test_cases)} test cases")

    # Initialize services
    chat_service = ChatService()
    evaluator = Evaluator()

    # Run all tests
    results = []
    eval_results = []

    for tc in test_cases:
        try:
            result = await run_single_test(chat_service, evaluator, tc, language)
            results.append(result)
            eval_results.append(result["eval_result"])
        except Exception as e:
            logger.error(f"Error on test {tc['id']}: {e}", exc_info=True)
            continue

    # Generate report
    report = EvaluationReport(results=eval_results)

    # Calculate aggregate retrieval metrics
    avg_hit_rate = sum(r["retrieval_metrics"]["hit_rate"] for r in results) / len(results) if results else 0
    avg_mrr = sum(r["retrieval_metrics"]["mrr"] for r in results) / len(results) if results else 0
    avg_p_at_k = sum(r["retrieval_metrics"]["precision_at_5"] for r in results) / len(results) if results else 0

    # Compile full results
    full_results = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "language": language,
            "total_tests": len(results),
            "test_cases_file": str(test_cases_path)
        },
        "summary": {
            "llm_judge_metrics": {
                "avg_faithfulness": round(report.avg_faithfulness, 3),
                "avg_relevance": round(report.avg_relevance, 3),
                "avg_citation": round(report.avg_citation, 3),
                "avg_overall": round(report.avg_overall, 3)
            },
            "retrieval_metrics": {
                "avg_hit_rate": round(avg_hit_rate, 3),
                "avg_mrr": round(avg_mrr, 3),
                "avg_precision_at_5": round(avg_p_at_k, 3)
            }
        },
        "detailed_results": results
    }

    # Print summary
    print("\n" + "=" * 70)
    print("CHATMYCV EVALUATION REPORT")
    print("=" * 70)
    print(f"Language: {language}")
    print(f"Total Tests: {len(results)}")
    print("-" * 70)
    print("LLM-AS-JUDGE METRICS:")
    print(f"  Faithfulness:  {report.avg_faithfulness:.2%}")
    print(f"  Relevance:     {report.avg_relevance:.2%}")
    print(f"  Citation:      {report.avg_citation:.2%}")
    print(f"  Overall:       {report.avg_overall:.2%}")
    print("-" * 70)
    print("RETRIEVAL METRICS:")
    print(f"  Hit Rate:      {avg_hit_rate:.2%}")
    print(f"  MRR:           {avg_mrr:.2%}")
    print(f"  Precision@5:   {avg_p_at_k:.2%}")
    print("=" * 70)

    # Save results
    if output_path is None:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        output_path = OUTPUT_DIR / f"eval_{language}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    with open(output_path, "w", encoding="utf-8") as f:
        # Convert EvaluationResult objects to dicts
        serializable_results = full_results.copy()
        serializable_results["detailed_results"] = [
            {
                **{k: v for k, v in r.items() if k != "eval_result"},
                "eval_result": {
                    "faithfulness_score": r["eval_result"].faithfulness_score,
                    "faithfulness_reason": r["eval_result"].faithfulness_reason,
                    "relevance_score": r["eval_result"].relevance_score,
                    "relevance_reason": r["eval_result"].relevance_reason,
                    "citation_score": r["eval_result"].citation_score,
                    "citation_reason": r["eval_result"].citation_reason,
                    "overall_score": r["eval_result"].overall_score
                }
            }
            for r in results
        ]
        json.dump(serializable_results, f, indent=2, ensure_ascii=False)

    logger.info(f"Results saved to: {output_path}")
    print(f"\nResults saved to: {output_path}")

    return full_results


def main():
    parser = argparse.ArgumentParser(description="Run ChatMyCV evaluation")
    parser.add_argument(
        "--language", "-l",
        choices=["en", "zhtw"],
        default="en",
        help="Language for evaluation (default: en)"
    )
    parser.add_argument(
        "--test-cases", "-t",
        type=Path,
        default=EVAL_DIR / "test_cases.json",
        help="Path to test cases JSON file"
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=None,
        help="Output path for results"
    )
    args = parser.parse_args()

    asyncio.run(run_evaluation(
        test_cases_path=args.test_cases,
        language=args.language,
        output_path=args.output
    ))


if __name__ == "__main__":
    main()
