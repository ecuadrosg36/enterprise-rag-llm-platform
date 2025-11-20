"""
Document processor pipeline.

Orchestrates document loading, chunking, and metadata management.
"""

from pathlib import Path
from typing import List, Union, Optional
from tqdm import tqdm

from .base_loader import Document
from .text_chunker import Chunk, RecursiveTextSplitter
from .pdf_loader import PDFLoader
from .txt_loader import TXTLoader
from .docx_loader import DOCXLoader
from src.core.errors import DocumentLoadError
from src.core.logger import setup_logger
from src.core.config import Config


logger = setup_logger(__name__)


class DocumentProcessor:
    """
    End-to-end document processing pipeline.
    
    Loads documents, chunks text, and manages metadata.
    """
    
    def __init__(self, config: Optional[Config] = None):
        """
        Initialize document processor.
        
        Args:
            config: Configuration object (if None, loads default)
        """
        if config is None:
            from src.core.config import get_config
            config = get_config()
        
        self.config = config
        
        # Initialize text chunker
        self.chunker = RecursiveTextSplitter(
            chunk_size=config.get('ingestion.chunk_size', 512),
            chunk_overlap=config.get('ingestion.chunk_overlap', 50),
            separators=config.get('ingestion.separators', ["\n\n", "\n", ". ", " "]),
            length_function="tokens"
        )
        
        logger.info(f"DocumentProcessor initialized (chunk_size={self.chunker.chunk_size})")
    
    def process_file(self, file_path: Union[str, Path]) -> List[Chunk]:
        """
        Process a single document file.
        
        Args:
            file_path: Path to document
            
        Returns:
            List of Chunk objects
        """
        file_path = Path(file_path)
        
        logger.info(f"Processing file: {file_path.name}")
        
        # Load document
        documents = self._load_document(file_path)
        
        if not documents:
            logger.warning(f"No content extracted from {file_path.name}")
            return []
        
        # Chunk documents
        all_chunks = []
        for doc in documents:
            chunks = self.chunker.split_text(doc.text, doc.metadata)
            all_chunks.extend(chunks)
        
        logger.info(f"Created {len(all_chunks)} chunks from {file_path.name}")
        return all_chunks
    
    def process_directory(
        self,
        directory: Union[str, Path],
        recursive: bool = True,
        show_progress: bool = True
    ) -> List[Chunk]:
        """
        Process all documents in a directory.
        
        Args:
            directory: Path to directory
            recursive: Process subdirectories
            show_progress: Show progress bar
            
        Returns:
            List of all chunks from all documents
        """
        directory = Path(directory)
        
        if not directory.exists():
            raise DocumentLoadError(
                f"Directory not found: {directory}",
                details={'path': str(directory)}
            )
        
        # Find all supported files
        supported_extensions = self.config.get('ingestion.supported_formats', ['pdf', 'txt', 'docx'])
        
        files = []
        for ext in supported_extensions:
            pattern = f"**/*.{ext}" if recursive else f"*.{ext}"
            files.extend(directory.glob(pattern))
        
        if not files:
            logger.warning(f"No supported documents found in {directory}")
            return []
        
        logger.info(f"Found {len(files)} documents in {directory}")
        
        # Process files
        all_chunks = []
        
        iterator = tqdm(files, desc="Processing documents") if show_progress else files
        
        for file_path in iterator:
            try:
                chunks = self.process_file(file_path)
                all_chunks.extend(chunks)
            except Exception as e:
                logger.error(f"Failed to process {file_path.name}: {e}")
                continue
        
        logger.info(f"Processed {len(files)} files → {len(all_chunks)} total chunks")
        return all_chunks
    
    def _load_document(self, file_path: Path) -> List[Document]:
        """Load document using appropriate loader."""
        extension = file_path.suffix.lower()
        
        try:
            if extension == '.pdf':
                loader = PDFLoader(file_path)
            elif extension == '.txt':
                loader = TXTLoader(file_path)
            elif extension in ['.docx', '.doc']:
                loader = DOCXLoader(file_path)
            else:
                raise DocumentLoadError(
                    f"Unsupported file type: {extension}",
                    details={'path': str(file_path), 'extension': extension}
                )
            
            return loader.load()
            
        except Exception as e:
            logger.error(f"Document loading failed: {file_path.name}", extra={'error': str(e)})
            raise
