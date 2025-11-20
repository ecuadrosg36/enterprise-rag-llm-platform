"""
Unit tests for document loaders and text chunker.
"""

import pytest
from pathlib import Path

from src.ingestion import (
    TXTLoader,
    RecursiveTextSplitter,
    DocumentProcessor,
    Chunk,
)
from src.core.errors import DocumentLoadError


class TestTXTLoader:
    """Tests for text file loader."""

    def test_load_txt_file(self, sample_txt_path):
        """Test loading a text file."""
        loader = TXTLoader(sample_txt_path)
        documents = loader.load()

        assert len(documents) == 1
        assert documents[0].text == "Sample text content for testing."
        assert documents[0].metadata["filename"] == "sample.txt"
        assert documents[0].metadata["file_type"] == ".txt"

    def test_load_nonexistent_file(self, temp_dir):
        """Test error on nonexistent file."""
        nonexistent = temp_dir / "nonexistent.txt"

        with pytest.raises(DocumentLoadError):
            TXTLoader(nonexistent)

    def test_empty_file(self, temp_dir):
        """Test handling of empty file."""
        empty_file = temp_dir / "empty.txt"
        empty_file.write_text("")

        loader = TXTLoader(empty_file)
        documents = loader.load()

        assert len(documents) == 0


class TestRecursiveTextSplitter:
    """Tests for recursive text splitter."""

    def test_split_short_text(self):
        """Test that short text is not split."""
        splitter = RecursiveTextSplitter(chunk_size=100, chunk_overlap=10)
        text = "This is a short text."

        chunks = splitter.split_text(text)

        assert len(chunks) == 1
        assert chunks[0].text == text

    def test_split_long_text(self):
        """Test splitting long text."""
        splitter = RecursiveTextSplitter(
            chunk_size=50, chunk_overlap=10, length_function="chars"
        )

        # Create text longer than chunk_size
        text = " ".join(["This is sentence number " + str(i) + "." for i in range(20)])

        chunks = splitter.split_text(text)

        assert len(chunks) > 1
        assert all(isinstance(chunk, Chunk) for chunk in chunks)

    def test_chunk_metadata(self):
        """Test that chunks have proper metadata."""
        splitter = RecursiveTextSplitter(
            chunk_size=50, chunk_overlap=10, length_function="chars"
        )
        text = " ".join(["Word"] * 100)

        chunks = splitter.split_text(text, metadata={"source": "test.txt"})

        assert all(chunk.metadata["source"] == "test.txt" for chunk in chunks)
        assert all("chunk_index" in chunk.metadata for chunk in chunks)
        assert all("total_chunks" in chunk.metadata for chunk in chunks)

    def test_empty_text(self):
        """Test handling of empty text."""
        splitter = RecursiveTextSplitter(chunk_size=100)

        chunks = splitter.split_text("")

        assert len(chunks) == 0

    def test_preserves_paragraph_boundaries(self):
        """Test that paragraph boundaries are preserved."""
        splitter = RecursiveTextSplitter(
            chunk_size=500, chunk_overlap=50, length_function="chars"
        )
        text = "Paragraph 1.\n\nParagraph 2.\n\nParagraph 3."

        chunks = splitter.split_text(text)

        # Should keep paragraphs together if possible
        assert len(chunks) == 1


class TestDocumentProcessor:
    """Tests for document processor."""

    def test_process_txt_file(self, sample_txt_path):
        """Test processing a text file."""
        processor = DocumentProcessor()
        chunks = processor.process_file(sample_txt_path)

        assert len(chunks) > 0
        assert all(isinstance(chunk, Chunk) for chunk in chunks)

    def test_process_unsupported_file(self, temp_dir):
        """Test error on unsupported file type."""
        unsupported = temp_dir / "test.xyz"
        unsupported.write_text("test")

        processor = DocumentProcessor()

        with pytest.raises(DocumentLoadError, match="Unsupported file type"):
            processor.process_file(unsupported)

    def test_process_directory(self, temp_dir):
        """Test processing a directory."""
        # Create multiple text files
        (temp_dir / "file1.txt").write_text("Content 1")
        (temp_dir / "file2.txt").write_text("Content 2")

        processor = DocumentProcessor()
        chunks = processor.process_directory(temp_dir, show_progress=False)

        assert len(chunks) >= 2  # At least one chunk per file

    def test_process_empty_directory(self, temp_dir):
        """Test processing empty directory."""
        processor = DocumentProcessor()
        chunks = processor.process_directory(temp_dir, show_progress=False)

        assert len(chunks) == 0
