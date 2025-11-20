"""
Synthetic dataset generator for RAG evaluation.

Uses LLMs to generate question-answer pairs from documents.
"""

from typing import List, Optional

try:
    from ragas.testset.generator import TestsetGenerator
    from ragas.testset.evolutions import simple, reasoning, multi_context
except ImportError:
    # Fallback for different ragas versions
    from ragas import TestsetGenerator

    simple = reasoning = multi_context = None
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import DirectoryLoader

from src.core.logger import setup_logger
from src.core.errors import EvaluationError


logger = setup_logger(__name__)


class DatasetGenerator:
    """
    Generates synthetic test datasets from documents.
    """

    def __init__(self, model_name: str = "gpt-3.5-turbo"):
        """
        Initialize generator.

        Args:
            model_name: OpenAI model to use for generation
        """
        self.generator_llm = ChatOpenAI(model=model_name)
        self.critic_llm = ChatOpenAI(
            model="gpt-4"
        )  # Critic usually needs stronger model
        self.embeddings = OpenAIEmbeddings()

        self.generator = TestsetGenerator.from_langchain(
            generator_llm=self.generator_llm,
            critic_llm=self.critic_llm,
            embeddings=self.embeddings,
        )

    def generate_from_directory(
        self,
        directory_path: str,
        test_size: int = 10,
        distributions: Optional[dict] = None,
    ) -> List[dict]:
        """
        Generate testset from documents in a directory.

        Args:
            directory_path: Path to documents
            test_size: Number of samples to generate
            distributions: Distribution of question types

        Returns:
            List of generated test samples
        """
        try:
            logger.info(f"Loading documents from {directory_path}...")
            loader = DirectoryLoader(directory_path)
            documents = loader.load()

            if not documents:
                raise EvaluationError(f"No documents found in {directory_path}")

            distributions = distributions or {
                simple: 0.5,
                reasoning: 0.25,
                multi_context: 0.25,
            }

            logger.info(f"Generating {test_size} test samples...")

            testset = self.generator.generate_with_langchain_docs(
                documents, test_size=test_size, distributions=distributions
            )

            logger.info("Dataset generation complete")
            return testset.to_pandas().to_dict(orient="records")

        except Exception as e:
            logger.error(f"Dataset generation failed: {e}")
            raise EvaluationError(f"Failed to generate dataset: {e}")
