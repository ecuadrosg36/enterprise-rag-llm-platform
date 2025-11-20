"""
RAG Evaluator using Ragas framework.

Computes metrics like faithfulness, answer relevancy, and context precision.
"""

import os
from typing import List, Dict, Any, Optional
import pandas as pd
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)

from src.core.logger import setup_logger
from src.core.errors import EvaluationError


logger = setup_logger(__name__)


class RAGEvaluator:
    """
    Evaluates RAG pipeline performance using Ragas metrics.
    """

    def __init__(self, metrics: Optional[List[Any]] = None):
        """
        Initialize evaluator.

        Args:
            metrics: List of Ragas metrics to use (defaults to standard set)
        """
        self.metrics = metrics or [
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
        ]

        # Check for OpenAI key (required for Ragas default models)
        if not os.getenv("OPENAI_API_KEY"):
            logger.warning("OPENAI_API_KEY not found. Ragas evaluation may fail.")

    def evaluate_results(
        self,
        questions: List[str],
        answers: List[str],
        contexts: List[List[str]],
        ground_truths: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Run evaluation on a set of results.

        Args:
            questions: List of user queries
            answers: List of generated answers
            contexts: List of retrieved contexts (list of strings per query)
            ground_truths: Optional list of expected answers (required for some metrics)

        Returns:
            Dictionary with aggregate scores and detailed results
        """
        try:
            # Prepare data for Ragas
            data = {
                "question": questions,
                "answer": answers,
                "contexts": contexts,
            }

            if ground_truths:
                data["ground_truth"] = ground_truths

            dataset = Dataset.from_dict(data)

            logger.info(f"Starting evaluation on {len(questions)} samples...")

            # Run evaluation
            results = evaluate(
                dataset=dataset,
                metrics=self.metrics,
            )

            logger.info("Evaluation complete")

            # Convert to dict
            return {
                "aggregate_scores": dict(results),
                "detailed_results": results.to_pandas().to_dict(orient="records"),
            }

        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            raise EvaluationError(f"Ragas evaluation failed: {e}")

    def evaluate_single(
        self,
        question: str,
        answer: str,
        contexts: List[str],
        ground_truth: Optional[str] = None,
    ) -> Dict[str, float]:
        """
        Evaluate a single result (wrapper around batch eval).
        """
        ground_truths = [ground_truth] if ground_truth else None

        results = self.evaluate_results(
            questions=[question],
            answers=[answer],
            contexts=[contexts],
            ground_truths=ground_truths,
        )

        return results["aggregate_scores"]
