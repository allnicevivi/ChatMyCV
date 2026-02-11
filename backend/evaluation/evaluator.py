"""
Prompt Evaluation System

Provides automated evaluation of RAG pipeline responses against
expected keywords and quality metrics. Used for regression testing
during prompt optimization.
"""

import json
import logging
import re
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


class PromptEvaluator:
    """
    Evaluator for RAG pipeline responses.

    Scores responses based on keyword matching and provides
    aggregate metrics for prompt performance.
    """

    def __init__(self):
        """Initialize the evaluator"""
        self.results = []

    def evaluate_answer(
        self,
        answer: str,
        expected_keywords: List[str],
        case_sensitive: bool = False
    ) -> Dict[str, Any]:
        """
        Evaluate a single answer against expected keywords.

        Args:
            answer: The generated answer text
            expected_keywords: List of keywords that should appear
            case_sensitive: Whether to use case-sensitive matching

        Returns:
            Dictionary with score and matching details
        """
        if not answer or answer == "None":
            return {
                "score": 0.0,
                "matched_keywords": [],
                "missing_keywords": expected_keywords,
                "match_count": 0,
                "total_keywords": len(expected_keywords)
            }

        # Prepare answer for matching
        answer_text = answer if case_sensitive else answer.lower()

        matched = []
        missing = []

        for keyword in expected_keywords:
            keyword_text = keyword if case_sensitive else keyword.lower()

            # Check if keyword appears in answer
            # Use word boundaries for English, simple substring for Chinese
            if re.search(r'[\u4e00-\u9fff]', keyword):
                # Chinese keyword - use substring match
                if keyword_text in answer_text:
                    matched.append(keyword)
                else:
                    missing.append(keyword)
            else:
                # English keyword - use word boundary match
                if re.search(rf'\b{re.escape(keyword_text)}\b', answer_text):
                    matched.append(keyword)
                else:
                    missing.append(keyword)

        # Calculate score: percentage of keywords matched
        score = len(matched) / len(expected_keywords) if expected_keywords else 0.0

        return {
            "score": score,
            "matched_keywords": matched,
            "missing_keywords": missing,
            "match_count": len(matched),
            "total_keywords": len(expected_keywords)
        }

    def run_evaluation(
        self,
        dataset_path: str,
        lang: str = "en",
        character: str = "hr",
        k: int = 5,
        temperature: float = 0.7,
        chat_service: Optional[Any] = None
    ) -> Dict[str, Any]:
        """
        Run evaluation on a dataset of questions.

        Args:
            dataset_path: Path to eval_dataset.json
            lang: Language ("en" or "zhtw")
            character: Interviewer character ("hr" or "engineer")
            k: Number of documents to retrieve
            temperature: LLM temperature
            chat_service: ChatService instance (for dependency injection)

        Returns:
            Evaluation results with aggregate metrics
        """
        # Import chat service if not provided
        if chat_service is None:
            import sys
            sys.path.append("./")
            sys.path.append("../")
            from services import chat_service as default_service
            chat_service = default_service

        # Load dataset
        try:
            with open(dataset_path, 'r', encoding='utf-8') as f:
                dataset = json.load(f)
        except Exception as e:
            logger.error(f"Failed to load dataset: {e}")
            return {"error": f"Failed to load dataset: {e}"}

        questions = dataset.get(lang, [])
        if not questions:
            return {"error": f"No questions found for language: {lang}"}

        logger.info(f"Running evaluation on {len(questions)} questions (lang={lang}, character={character})")

        results = []
        total_score = 0.0
        total_time = 0.0

        for i, item in enumerate(questions, 1):
            question = item.get("question")
            expected_keywords = item.get("expected_keywords", [])
            category = item.get("category", "general")

            logger.info(f"[{i}/{len(questions)}] Evaluating: {question[:50]}...")

            try:
                # Run the question through the RAG pipeline
                start_time = time.time()

                response = chat_service.chat(
                    lang=lang,
                    query=question,
                    session_id=None,  # Don't persist evaluation queries
                    k=k,
                    temperature=temperature,
                    character=character
                )

                elapsed_time = time.time() - start_time

                answer = response.get("content")

                # Evaluate the answer
                eval_result = self.evaluate_answer(answer, expected_keywords)

                # Compile result
                result = {
                    "question": question,
                    "answer": answer,
                    "expected_keywords": expected_keywords,
                    "category": category,
                    "score": eval_result["score"],
                    "matched_keywords": eval_result["matched_keywords"],
                    "missing_keywords": eval_result["missing_keywords"],
                    "elapsed_time": elapsed_time,
                    "retrieved_docs_count": response.get("retrieved_docs_count", 0),
                    "avg_similarity": response.get("avg_similarity", 0.0),
                    "trace_id": response.get("trace_id")
                }

                results.append(result)
                total_score += eval_result["score"]
                total_time += elapsed_time

                logger.info(f"  Score: {eval_result['score']:.2f} | Time: {elapsed_time:.2f}s")

            except Exception as e:
                logger.error(f"Error evaluating question: {e}", exc_info=True)
                results.append({
                    "question": question,
                    "error": str(e),
                    "score": 0.0
                })

        # Calculate aggregate metrics
        avg_score = total_score / len(questions) if questions else 0.0
        avg_time = total_time / len(questions) if questions else 0.0

        # Category breakdown
        category_scores = {}
        for result in results:
            cat = result.get("category", "general")
            if cat not in category_scores:
                category_scores[cat] = []
            if "score" in result:
                category_scores[cat].append(result["score"])

        category_avg = {
            cat: sum(scores) / len(scores) if scores else 0.0
            for cat, scores in category_scores.items()
        }

        # Summary
        summary = {
            "total_questions": len(questions),
            "avg_score": avg_score,
            "avg_time": avg_time,
            "total_time": total_time,
            "category_scores": category_avg,
            "pass_threshold": 0.6,  # 60% keyword match
            "passed": avg_score >= 0.6,
            "timestamp": datetime.now().isoformat(),
            "config": {
                "lang": lang,
                "character": character,
                "k": k,
                "temperature": temperature
            }
        }

        evaluation_result = {
            "summary": summary,
            "results": results
        }

        self.results = results

        logger.info(f"Evaluation complete. Average score: {avg_score:.2%}")

        return evaluation_result

    def save_results(self, output_path: str, results: Dict[str, Any]) -> bool:
        """
        Save evaluation results to JSON file.

        Args:
            output_path: Path to output file
            results: Evaluation results dictionary

        Returns:
            True if successful
        """
        try:
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)

            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)

            logger.info(f"Results saved to: {output_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to save results: {e}")
            return False

    def print_summary(self, results: Dict[str, Any]) -> None:
        """
        Print a formatted summary of evaluation results.

        Args:
            results: Evaluation results dictionary
        """
        summary = results.get("summary", {})

        print("\n" + "="*60)
        print("EVALUATION SUMMARY")
        print("="*60)
        print(f"Total Questions:    {summary.get('total_questions', 0)}")
        print(f"Average Score:      {summary.get('avg_score', 0.0):.2%}")
        print(f"Average Time:       {summary.get('avg_time', 0.0):.2f}s")
        print(f"Total Time:         {summary.get('total_time', 0.0):.2f}s")
        print(f"Pass Threshold:     {summary.get('pass_threshold', 0.6):.2%}")
        print(f"Result:             {'PASSED ✓' if summary.get('passed') else 'FAILED ✗'}")
        print("\nCategory Breakdown:")
        for category, score in summary.get('category_scores', {}).items():
            print(f"  {category:15s} {score:.2%}")
        print("="*60 + "\n")


# Export
__all__ = [
    'PromptEvaluator'
]
