#!/usr/bin/env python
# -*- coding:utf-8 -*-

"""
Custom Evaluation Module for ChatMyCV

Provides LLM-as-Judge evaluation for:
- Faithfulness: Is the answer grounded in the retrieved context?
- Relevance: Does the answer address the question?
- Citation Accuracy: Are sources properly attributed?
"""

import sys
sys.path.append("./")
sys.path.append("../")

import json
import asyncio
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional
from backend.llm.azure_module import AzureOpenaiLLM
from utils.app_logger import LoggerSetup

logger = LoggerSetup("Evaluator").logger


@dataclass
class EvaluationResult:
    """Single evaluation result."""
    test_id: str
    question: str
    answer: str
    context: str
    faithfulness_score: float
    faithfulness_reason: str
    relevance_score: float
    relevance_reason: str
    citation_score: float
    citation_reason: str

    @property
    def overall_score(self) -> float:
        """Weighted average of all scores."""
        return (
            self.faithfulness_score * 0.4 +
            self.relevance_score * 0.4 +
            self.citation_score * 0.2
        )


FAITHFULNESS_PROMPT = """You are an expert evaluator assessing whether an AI answer is faithful to the provided context.

**Context (Retrieved from CV):**
{context}

**Question:**
{question}

**Answer:**
{answer}

**Task:**
Evaluate if the answer is factually consistent with the context. The answer should:
1. Only contain information that can be verified from the context
2. Not hallucinate or make up information not present in the context
3. Accurately represent the information from the context

**Scoring:**
- 1.0: Completely faithful, all claims supported by context
- 0.75: Mostly faithful, minor unsupported details
- 0.5: Partially faithful, some claims not in context
- 0.25: Mostly unfaithful, significant hallucinations
- 0.0: Completely unfaithful or contradicts context

Respond in JSON format:
{{"score": <float>, "reason": "<brief explanation>"}}"""


RELEVANCE_PROMPT = """You are an expert evaluator assessing whether an AI answer is relevant to the question.

**Question:**
{question}

**Answer:**
{answer}

**Task:**
Evaluate if the answer directly addresses the question asked. The answer should:
1. Directly respond to what was asked
2. Be on-topic and focused
3. Provide useful information for the question

**Scoring:**
- 1.0: Highly relevant, directly answers the question
- 0.75: Mostly relevant, addresses the question with minor tangents
- 0.5: Partially relevant, some useful information but misses key points
- 0.25: Marginally relevant, mostly off-topic
- 0.0: Not relevant at all

Respond in JSON format:
{{"score": <float>, "reason": "<brief explanation>"}}"""


CITATION_PROMPT = """You are an expert evaluator assessing citation accuracy in an AI answer.

**Context (Retrieved from CV):**
{context}

**Expected Source Sections:**
{expected_sources}

**Answer:**
{answer}

**Task:**
Evaluate if the answer properly uses information from the expected source sections. Consider:
1. Does the answer draw from the relevant sections of the CV?
2. Are the sources of information traceable to the context?
3. If information is not available, does the answer acknowledge this?

**Scoring:**
- 1.0: Excellent citation, clearly uses correct sources
- 0.75: Good citation, mostly from expected sources
- 0.5: Partial citation, some information from unexpected sources
- 0.25: Poor citation, sources unclear or incorrect
- 0.0: No citation or completely wrong sources

Respond in JSON format:
{{"score": <float>, "reason": "<brief explanation>"}}"""


class Evaluator:
    """LLM-as-Judge evaluator for RAG responses."""

    def __init__(self, llm: Optional[AzureOpenaiLLM] = None):
        self.llm = llm or AzureOpenaiLLM()

    async def _judge(self, prompt: str) -> Dict[str, Any]:
        """Call LLM to judge and parse JSON response."""
        try:
            response = await self.llm.chat(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0
            )
            # Extract JSON from response
            content = response.strip()
            if content.startswith("```"):
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]
            return json.loads(content)
        except Exception as e:
            logger.error(f"Judge parsing error: {e}")
            return {"score": 0.0, "reason": f"Evaluation error: {str(e)}"}

    async def evaluate_faithfulness(
        self,
        question: str,
        answer: str,
        context: str
    ) -> Dict[str, Any]:
        """Evaluate if answer is faithful to context."""
        prompt = FAITHFULNESS_PROMPT.format(
            context=context,
            question=question,
            answer=answer
        )
        return await self._judge(prompt)

    async def evaluate_relevance(
        self,
        question: str,
        answer: str
    ) -> Dict[str, Any]:
        """Evaluate if answer is relevant to question."""
        prompt = RELEVANCE_PROMPT.format(
            question=question,
            answer=answer
        )
        return await self._judge(prompt)

    async def evaluate_citation(
        self,
        answer: str,
        context: str,
        expected_sources: List[str]
    ) -> Dict[str, Any]:
        """Evaluate citation accuracy."""
        prompt = CITATION_PROMPT.format(
            context=context,
            expected_sources=", ".join(expected_sources) if expected_sources else "N/A",
            answer=answer
        )
        return await self._judge(prompt)

    async def evaluate(
        self,
        test_id: str,
        question: str,
        answer: str,
        context: str,
        expected_sources: List[str] = None
    ) -> EvaluationResult:
        """Run full evaluation on a single Q&A pair."""
        expected_sources = expected_sources or []

        # Run all evaluations in parallel
        faithfulness, relevance, citation = await asyncio.gather(
            self.evaluate_faithfulness(question, answer, context),
            self.evaluate_relevance(question, answer),
            self.evaluate_citation(answer, context, expected_sources)
        )

        return EvaluationResult(
            test_id=test_id,
            question=question,
            answer=answer,
            context=context[:500] + "..." if len(context) > 500 else context,
            faithfulness_score=faithfulness.get("score", 0.0),
            faithfulness_reason=faithfulness.get("reason", ""),
            relevance_score=relevance.get("score", 0.0),
            relevance_reason=relevance.get("reason", ""),
            citation_score=citation.get("score", 0.0),
            citation_reason=citation.get("reason", "")
        )


@dataclass
class EvaluationReport:
    """Aggregated evaluation report."""
    results: List[EvaluationResult]

    @property
    def avg_faithfulness(self) -> float:
        scores = [r.faithfulness_score for r in self.results]
        return sum(scores) / len(scores) if scores else 0.0

    @property
    def avg_relevance(self) -> float:
        scores = [r.relevance_score for r in self.results]
        return sum(scores) / len(scores) if scores else 0.0

    @property
    def avg_citation(self) -> float:
        scores = [r.citation_score for r in self.results]
        return sum(scores) / len(scores) if scores else 0.0

    @property
    def avg_overall(self) -> float:
        scores = [r.overall_score for r in self.results]
        return sum(scores) / len(scores) if scores else 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "summary": {
                "total_tests": len(self.results),
                "avg_faithfulness": round(self.avg_faithfulness, 3),
                "avg_relevance": round(self.avg_relevance, 3),
                "avg_citation": round(self.avg_citation, 3),
                "avg_overall": round(self.avg_overall, 3)
            },
            "results": [asdict(r) for r in self.results]
        }

    def print_summary(self):
        print("\n" + "=" * 60)
        print("EVALUATION REPORT")
        print("=" * 60)
        print(f"Total Test Cases: {len(self.results)}")
        print(f"Avg Faithfulness: {self.avg_faithfulness:.2%}")
        print(f"Avg Relevance:    {self.avg_relevance:.2%}")
        print(f"Avg Citation:     {self.avg_citation:.2%}")
        print(f"Avg Overall:      {self.avg_overall:.2%}")
        print("=" * 60)

        for r in self.results:
            print(f"\n[{r.test_id}] {r.question[:50]}...")
            print(f"  Faithfulness: {r.faithfulness_score:.2f} - {r.faithfulness_reason[:60]}")
            print(f"  Relevance:    {r.relevance_score:.2f} - {r.relevance_reason[:60]}")
            print(f"  Citation:     {r.citation_score:.2f} - {r.citation_reason[:60]}")
            print(f"  Overall:      {r.overall_score:.2f}")


if __name__ == "__main__":
    # Quick test
    async def test():
        evaluator = Evaluator()
        result = await evaluator.evaluate(
            test_id="test_001",
            question="What is the candidate's current role?",
            answer="The candidate is currently a Senior Software Engineer at TechCorp.",
            context="Work Experience: Senior Software Engineer at TechCorp (2022-present). Responsible for backend development.",
            expected_sources=["Work Experience"]
        )
        print(f"Faithfulness: {result.faithfulness_score}")
        print(f"Relevance: {result.relevance_score}")
        print(f"Citation: {result.citation_score}")
        print(f"Overall: {result.overall_score}")

    asyncio.run(test())
