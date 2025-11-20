"""
Ingestion Script Example.

Demonstrates how to ingest documents into the RAG platform.
"""

import os
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.core.config import get_config
from src.core.logger import setup_logger
from src.ingestion import DocumentProcessor
from src.embeddings import EmbeddingFactory
from src.vector_store import IndexManager, ChromaVectorStore

logger = setup_logger(__name__)

def main():
    # 1. Load Config
    config = get_config()
    logger.info("Starting ingestion process...")
    
    # 2. Initialize Components
    # Processor: Loads and chunks documents
    processor = DocumentProcessor(
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap
    )
    
    # Embedder: Creates vector embeddings
    embedder = EmbeddingFactory.create(config)
    
    # Vector Store: Persists vectors
    vector_store = ChromaVectorStore(config)
    
    # Index Manager: Orchestrates indexing
    index_manager = IndexManager(vector_store, embedder)
    
    # 3. Process Documents
    data_dir = Path("data")
    if not data_dir.exists():
        logger.error(f"Data directory not found: {data_dir}")
        return
        
    logger.info(f"Processing documents from: {data_dir}")
    chunks = processor.process_directory(str(data_dir))
    logger.info(f"Generated {len(chunks)} chunks")
    
    if not chunks:
        logger.warning("No chunks to index.")
        return
        
    # 4. Index Chunks
    logger.info("Indexing chunks...")
    index_manager.index_batch(chunks, batch_size=50)
    
    logger.info("Ingestion complete! 🚀")

if __name__ == "__main__":
    main()
