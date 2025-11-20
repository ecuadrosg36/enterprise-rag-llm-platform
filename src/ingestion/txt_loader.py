"""
Plain text document loader.
"""

from pathlib import Path
from typing import List

from .base_loader import BaseDocumentLoader, Document
from src.core.errors import DocumentLoadError
from src.core.logger import setup_logger


logger = setup_logger(__name__)


class TXTLoader(BaseDocumentLoader):
    """Load plain text documents."""
    
    def __init__(self, file_path: Path, encoding: str = 'utf-8'):
        """
        Initialize TXT loader.
        
        Args:
            file_path: Path to text file
            encoding: Text encoding (default: utf-8)
        """
        super().__init__(file_path)
        self.encoding = encoding
    
    def load(self) -> List[Document]:
        """
        Load text file.
        
        Returns:
            List containing single Document object
        """
        try:
            with open(self.file_path, 'r', encoding=self.encoding) as f:
                text = f.read()
            
            if not text.strip():
                logger.warning(f"Empty text file: {self.file_path.name}")
                return []
            
            metadata = self._get_base_metadata()
            metadata['encoding'] = self.encoding
            metadata['line_count'] = text.count('\n') + 1
            metadata['char_count'] = len(text)
            
            document = Document(
                text=text.strip(),
                metadata=metadata
            )
            
            logger.info(f"Loaded text file: {self.file_path.name} ({len(text)} chars)")
            return [document]
            
        except UnicodeDecodeError as e:
            logger.error(f"Encoding error loading {self.file_path.name}: {e}")
            raise DocumentLoadError(
                f"Failed to decode text file with encoding {self.encoding}",
                details={'path': str(self.file_path), 'encoding': self.encoding}
            )
        except Exception as e:
            logger.error(f"Failed to load text file: {self.file_path.name}", extra={'error': str(e)})
            raise DocumentLoadError(
                f"Text file loading failed: {e}",
                details={'path': str(self.file_path), 'error': str(e)}
            )
