"""
Pytest configuration and shared fixtures.
"""

import pytest
import tempfile
from pathlib import Path


@pytest.fixture
def temp_dir():
    """Temporary directory for tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_text():
    """Sample text for testing chunking."""
    return """
    This is a sample document for testing.
    
    It has multiple paragraphs. Each paragraph contains several sentences.
    This helps test the text splitter.
    
    The document should be split into meaningful chunks.
    """.strip()


@pytest.fixture
def sample_pdf_path(temp_dir):
    """Path for sample PDF file."""
    return temp_dir / "sample.pdf"


@pytest.fixture
def sample_txt_path(temp_dir):
    """Path for sample TXT file."""
    txt_path = temp_dir / "sample.txt"
    txt_path.write_text("Sample text content for testing.")
    return txt_path
