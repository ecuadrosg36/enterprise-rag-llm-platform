"""
Base document loader interface.

All document loaders inherit from BaseDocumentLoader and implement
the load() method to extract text and metadata from documents.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Dict, Any
from dataclasses import dataclass
from datetime import datetime


@dataclass
class Document:
    """Container for document text and metadata."""

    text: str
    metadata: Dict[str, Any]

    def __repr__(self) -> str:
        preview = self.text[:50] + "..." if len(self.text) > 50 else self.text
        return f"Document(text='{preview}', metadata={self.metadata})"


class BaseDocumentLoader(ABC):
    """Abstract base class for document loaders."""

    def __init__(self, file_path: Path):
        """
        Initialize document loader.

        Args:
            file_path: Path to document file
        """
        self.file_path = Path(file_path)

        if not self.file_path.exists():
            from src.core.errors import DocumentLoadError

            raise DocumentLoadError(
                f"File not found: {file_path}", details={"path": str(file_path)}
            )

    @abstractmethod
    def load(self) -> List[Document]:
        """
        Load and parse document.

        Returns:
            List of Document objects (one per page/section)
        """
        pass

    def _get_base_metadata(self) -> Dict[str, Any]:
        """Get common metadata for all documents."""
        stat = self.file_path.stat()

        return {
            "source": str(self.file_path),
            "filename": self.file_path.name,
            "file_type": self.file_path.suffix.lower(),
            "file_size": stat.st_size,
            "created_at": datetime.fromtimestamp(stat.st_ctime).isoformat(),
            "modified_at": datetime.fromtimestamp(stat.st_mtime).isoformat(),
        }
