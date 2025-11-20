"""
Recursive text splitter for chunking documents.

Implements recursive character-based text splitting with overlap,
preserving semantic boundaries (paragraphs, sentences, words).
"""

from typing import List, Dict, Any
from dataclasses import dataclass
import tiktoken

from src.core.errors import ChunkingError
from src.core.logger import setup_logger


logger = setup_logger(__name__)


@dataclass
class Chunk:
    """Container for text chunk with metadata."""

    text: str
    metadata: Dict[str, Any]

    def __repr__(self) -> str:
        preview = self.text[:50] + "..." if len(self.text) > 50 else self.text
        return f"Chunk(text='{preview}', metadata={self.metadata})"


class RecursiveTextSplitter:
    """
    Split text recursively using a hierarchy of separators.

    Tries to split by paragraphs first, then sentences, then words,
    preserving semantic boundaries as much as possible.
    """

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        separators: List[str] = None,
        length_function: str = "tokens",  # "tokens" or "chars"
    ):
        """
        Initialize text splitter.

        Args:
            chunk_size: Target chunk size (in tokens or characters)
            chunk_overlap: Overlap between chunks (in tokens or characters)
            separators: List of separators to try (hierarchical)
            length_function: How to measure length ("tokens" or "chars")
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or ["\n\n", "\n", ". ", " ", ""]
        self.length_function = length_function

        # Initialize tokenizer for token counting
        if length_function == "tokens":
            try:
                self.tokenizer = tiktoken.get_encoding("cl100k_base")
            except:
                logger.warning(
                    "Failed to load tiktoken, falling back to character length"
                )
                self.length_function = "chars"

    def split_text(self, text: str, metadata: Dict[str, Any] = None) -> List[Chunk]:
        """
        Split text into chunks.

        Args:
            text: Text to split
            metadata: Base metadata to attach to all chunks

        Returns:
            List of Chunk objects
        """
        if not text or not text.strip():
            return []

        metadata = metadata or {}

        try:
            chunks = self._split_text_recursive(text.strip())

            # Create Chunk objects with metadata
            result = []
            for i, chunk_text in enumerate(chunks):
                chunk_metadata = metadata.copy()
                chunk_metadata["chunk_index"] = i
                chunk_metadata["total_chunks"] = len(chunks)
                chunk_metadata["chunk_size"] = self._get_length(chunk_text)

                result.append(Chunk(text=chunk_text, metadata=chunk_metadata))

            logger.info(f"Split text into {len(result)} chunks")
            return result

        except Exception as e:
            logger.error(f"Text splitting failed: {e}")
            raise ChunkingError(
                f"Failed to split text: {e}", details={"text_length": len(text)}
            )

    def _split_text_recursive(self, text: str) -> List[str]:
        """Recursively split text using separator hierarchy."""
        # Base case: text is small enough
        if self._get_length(text) <= self.chunk_size:
            return [text] if text.strip() else []

        # Try each separator
        for separator in self.separators:
            if separator in text:
                chunks = []
                splits = text.split(separator)

                current_chunk = ""
                for split in splits:
                    # Add separator back (except for empty separator)
                    split_with_sep = split + (
                        separator if separator and split != splits[-1] else ""
                    )

                    test_chunk = current_chunk + split_with_sep

                    if self._get_length(test_chunk) <= self.chunk_size:
                        current_chunk = test_chunk
                    else:
                        # Current chunk is full, save it
                        if current_chunk:
                            chunks.append(current_chunk.strip())

                        # Start new chunk
                        if self._get_length(split_with_sep) > self.chunk_size:
                            # Split is too large, recursively split it
                            sub_chunks = self._split_text_recursive(split_with_sep)
                            chunks.extend(sub_chunks[:-1])
                            current_chunk = sub_chunks[-1] if sub_chunks else ""
                        else:
                            current_chunk = split_with_sep

                # Add remaining chunk
                if current_chunk.strip():
                    chunks.append(current_chunk.strip())

                # Apply overlap
                return self._apply_overlap(chunks)

        # Fallback: split by chunk_size
        return self._split_by_size(text)

    def _apply_overlap(self, chunks: List[str]) -> List[str]:
        """Add overlap between consecutive chunks."""
        if len(chunks) <= 1 or self.chunk_overlap == 0:
            return chunks

        overlapped = []

        for i, chunk in enumerate(chunks):
            if i == 0:
                overlapped.append(chunk)
            else:
                # Get overlap from previous chunk
                prev_chunk = chunks[i - 1]
                overlap_text = self._get_overlap_text(prev_chunk, self.chunk_overlap)

                # Prepend overlap to current chunk
                overlapped.append(overlap_text + chunk)

        return overlapped

    def _get_overlap_text(self, text: str, overlap_size: int) -> str:
        """Get the last overlap_size tokens/chars from text."""
        if self.length_function == "tokens":
            tokens = self.tokenizer.encode(text)
            if len(tokens) <= overlap_size:
                return text
            overlap_tokens = tokens[-overlap_size:]
            return self.tokenizer.decode(overlap_tokens)
        else:
            return text[-overlap_size:] if len(text) > overlap_size else text

    def _split_by_size(self, text: str) -> List[str]:
        """Split text by fixed size (fallback method)."""
        chunks = []

        if self.length_function == "tokens":
            tokens = self.tokenizer.encode(text)
            for i in range(0, len(tokens), self.chunk_size - self.chunk_overlap):
                chunk_tokens = tokens[i : i + self.chunk_size]
                chunks.append(self.tokenizer.decode(chunk_tokens))
        else:
            for i in range(0, len(text), self.chunk_size - self.chunk_overlap):
                chunks.append(text[i : i + self.chunk_size])

        return chunks

    def _get_length(self, text: str) -> int:
        """Get length of text (tokens or characters)."""
        if self.length_function == "tokens":
            return len(self.tokenizer.encode(text))
        else:
            return len(text)
