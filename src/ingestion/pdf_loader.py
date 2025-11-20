"""
PDF document loader using PyPDF2 and pdfplumber.

Supports both PyPDF2 (faster) and pdfplumber (better for complex PDFs).
"""

from pathlib import Path
from typing import List
import PyPDF2
import pdfplumber

from .base_loader import BaseDocumentLoader, Document
from src.core.errors import DocumentLoadError
from src.core.logger import setup_logger


logger = setup_logger(__name__)


class PDFLoader(BaseDocumentLoader):
    """Load PDF documents and extract text by page."""

    def __init__(self, file_path: Path, use_pdfplumber: bool = False):
        """
        Initialize PDF loader.

        Args:
            file_path: Path to PDF file
            use_pdfplumber: Use pdfplumber instead of PyPDF2 (slower but better)
        """
        super().__init__(file_path)
        self.use_pdfplumber = use_pdfplumber

    def load(self) -> List[Document]:
        """
        Load PDF and extract text from each page.

        Returns:
            List of Document objects (one per page)
        """
        try:
            if self.use_pdfplumber:
                return self._load_with_pdfplumber()
            else:
                return self._load_with_pypdf2()
        except Exception as e:
            logger.error(
                f"Failed to load PDF: {self.file_path}", extra={"error": str(e)}
            )
            raise DocumentLoadError(
                f"PDF loading failed: {e}",
                details={"path": str(self.file_path), "error": str(e)},
            )

    def _load_with_pypdf2(self) -> List[Document]:
        """Load with PyPDF2 (faster)."""
        documents = []
        base_metadata = self._get_base_metadata()

        with open(self.file_path, "rb") as f:
            pdf_reader = PyPDF2.PdfReader(f)
            total_pages = len(pdf_reader.pages)

            base_metadata["total_pages"] = total_pages

            for page_num in range(total_pages):
                page = pdf_reader.pages[page_num]
                text = page.extract_text()

                if text and text.strip():
                    metadata = base_metadata.copy()
                    metadata["page"] = page_num + 1  # 1-indexed

                    documents.append(Document(text=text.strip(), metadata=metadata))

        logger.info(f"Loaded {len(documents)} pages from PDF: {self.file_path.name}")
        return documents

    def _load_with_pdfplumber(self) -> List[Document]:
        """Load with pdfplumber (better text extraction)."""
        documents = []
        base_metadata = self._get_base_metadata()

        with pdfplumber.open(self.file_path) as pdf:
            total_pages = len(pdf.pages)
            base_metadata["total_pages"] = total_pages

            for page_num, page in enumerate(pdf.pages):
                text = page.extract_text()

                if text and text.strip():
                    metadata = base_metadata.copy()
                    metadata["page"] = page_num + 1

                    # Add page dimensions (useful for layout analysis)
                    metadata["page_width"] = page.width
                    metadata["page_height"] = page.height

                    documents.append(Document(text=text.strip(), metadata=metadata))

        logger.info(
            f"Loaded {len(documents)} pages from PDF (pdfplumber): {self.file_path.name}"
        )
        return documents
