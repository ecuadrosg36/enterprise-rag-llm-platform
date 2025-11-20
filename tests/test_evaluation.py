"""
Unit tests for evaluation layer.
"""

import pytest
from unittest.mock import Mock, patch
import pandas as pd

from src.evaluation import RAGEvaluator
from src.core.errors import EvaluationError


class TestRAGEvaluator:
    """Tests for RAG Evaluator."""
    
    @patch("src.evaluation.evaluator.evaluate")
    def test_evaluate_results(self, mock_evaluate):
        """Test evaluation flow."""
        # Mock Ragas evaluate return
        mock_results = Mock()
        mock_results.items.return_value = [("faithfulness", 0.9)]
        mock_results.__iter__ = Mock(return_value=iter([("faithfulness", 0.9)]))
        # Mock to_pandas
        mock_df = pd.DataFrame([{"faithfulness": 0.9, "question": "q1"}])
        mock_results.to_pandas.return_value = mock_df
        
        mock_evaluate.return_value = mock_results
        
        evaluator = RAGEvaluator(metrics=[Mock()])
        
        results = evaluator.evaluate_results(
            questions=["q1"],
            answers=["a1"],
            contexts=[["c1"]]
        )
        
        assert "aggregate_scores" in results
        assert "detailed_results" in results
        assert results["aggregate_scores"]["faithfulness"] == 0.9
        
        mock_evaluate.assert_called_once()
    
    def test_initialization_warning(self):
        """Test warning if API key missing."""
        with patch.dict("os.environ", {}, clear=True):
            with patch("src.core.logger.logger.warning") as mock_warn:
                RAGEvaluator()
                mock_warn.assert_called_with("OPENAI_API_KEY not found. Ragas evaluation may fail.")
