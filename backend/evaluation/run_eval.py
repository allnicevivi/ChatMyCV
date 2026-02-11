#!/usr/bin/env python
# -*- coding:utf-8 -*-

"""
Prompt Evaluation CLI

Command-line interface for running prompt evaluations against test dataset.

Usage:
    python backend/evaluation/run_eval.py --lang en --character hr
    python backend/evaluation/run_eval.py --lang zhtw --character engineer --k 3
    python backend/evaluation/run_eval.py --lang en --output results/eval_$(date +%Y%m%d).json
"""

import argparse
import sys
import os
from pathlib import Path

# Add backend directory to path
sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent.parent.parent))

from evaluator import PromptEvaluator
from services import chat_service
from utils.app_logger import LoggerSetup

logger = LoggerSetup("EvalCLI").logger


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description="Run prompt evaluation on ChatMyCV RAG system",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--lang",
        type=str,
        choices=["en", "zhtw"],
        default="en",
        help="Language for evaluation dataset"
    )

    parser.add_argument(
        "--character",
        type=str,
        choices=["hr", "engineer", "engineering", "eng"],
        default="hr",
        help="Interviewer character/persona"
    )

    parser.add_argument(
        "--k",
        type=int,
        default=5,
        help="Number of documents to retrieve from vector store"
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="LLM sampling temperature"
    )

    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Path to evaluation dataset JSON file (default: backend/evaluation/eval_dataset.json)"
    )

    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save evaluation results (default: backend/evaluation/results/eval_<timestamp>.json)"
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )

    args = parser.parse_args()

    # Normalize character name
    character = args.character
    if character in ["engineering", "eng"]:
        character = "engineer"

    # Determine dataset path
    if args.dataset:
        dataset_path = args.dataset
    else:
        # Default to eval_dataset.json in same directory
        dataset_path = Path(__file__).parent / "eval_dataset.json"

    if not Path(dataset_path).exists():
        logger.error(f"Dataset not found: {dataset_path}")
        sys.exit(1)

    # Determine output path
    if args.output:
        output_path = args.output
    else:
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_dir = Path(__file__).parent / "results"
        results_dir.mkdir(exist_ok=True)
        output_path = results_dir / f"eval_{args.lang}_{character}_{timestamp}.json"

    # Print configuration
    print("\n" + "="*60)
    print("PROMPT EVALUATION")
    print("="*60)
    print(f"Dataset:      {dataset_path}")
    print(f"Language:     {args.lang}")
    print(f"Character:    {character}")
    print(f"K (retrieval): {args.k}")
    print(f"Temperature:  {args.temperature}")
    print(f"Output:       {output_path}")
    print("="*60 + "\n")

    # Create evaluator
    evaluator = PromptEvaluator()

    # Run evaluation
    try:
        logger.info("Starting evaluation...")
        results = evaluator.run_evaluation(
            dataset_path=str(dataset_path),
            lang=args.lang,
            character=character,
            k=args.k,
            temperature=args.temperature,
            chat_service=chat_service
        )

        if "error" in results:
            logger.error(f"Evaluation failed: {results['error']}")
            sys.exit(1)

        # Print summary
        evaluator.print_summary(results)

        # Save results
        evaluator.save_results(str(output_path), results)

        # Exit with appropriate code
        passed = results.get("summary", {}).get("passed", False)
        sys.exit(0 if passed else 1)

    except KeyboardInterrupt:
        logger.warning("Evaluation interrupted by user")
        sys.exit(130)

    except Exception as e:
        logger.error(f"Evaluation failed with error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
