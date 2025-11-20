# Evaluation package
from .evaluator import RAGEvaluator

try:
    from .dataset_generator import DatasetGenerator

    __all__ = ["RAGEvaluator", "DatasetGenerator"]
except ImportError:
    # DatasetGenerator requires ragas which may not be available
    __all__ = ["RAGEvaluator"]
