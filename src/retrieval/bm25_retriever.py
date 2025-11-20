"""
BM25 Keyword Retriever.

Implements sparse keyword search using the rank-bm25 library.
Useful for exact match queries where semantic search might fail.
"""

from typing import List, Dict, Any, Optional
import pickle
from pathlib import Path
from rank_bm25 import BM25Okapi
from tqdm import tqdm

from .base_retriever import BaseRetriever
from src.vector_store.base_store import SearchResult
from src.ingestion.base_loader import Document
from src.ingestion.text_chunker import Chunk
from src.core.logger import setup_logger
from src.core.errors import RetrievalError


logger = setup_logger(__name__)


class BM25Retriever(BaseRetriever):
    """
    BM25 retriever for keyword-based search.
    
    Maintains an in-memory index of tokenized documents.
    """
    
    def __init__(self, persist_dir: Optional[Path] = None):
        """
        Initialize BM25 retriever.
        
        Args:
            persist_dir: Directory to save/load index
        """
        self.bm25: Optional[BM25Okapi] = None
        self.documents: List[Document] = []
        self.persist_dir = persist_dir
        
        if persist_dir:
            self.persist_dir.mkdir(parents=True, exist_ok=True)
            self.index_path = self.persist_dir / "bm25_index.pkl"
            self._load_index()
    
    def index_documents(self, chunks: List[Chunk]):
        """
        Build BM25 index from chunks.
        
        Args:
            chunks: List of text chunks to index
        """
        if not chunks:
            logger.warning("No chunks provided for BM25 indexing")
            return
        
        logger.info(f"Building BM25 index for {len(chunks)} documents...")
        
        # Store documents for retrieval
        self.documents = [
            Document(text=c.text, metadata=c.metadata) 
            for c in chunks
        ]
        
        # Tokenize corpus (simple whitespace tokenization for now)
        # In production, use a proper tokenizer (spacy/nltk)
        tokenized_corpus = [doc.text.lower().split() for doc in self.documents]
        
        # Build index
        self.bm25 = BM25Okapi(tokenized_corpus)
        
        logger.info("BM25 index built successfully")
        self._save_index()
    
    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """
        Retrieve documents using BM25.
        
        Args:
            query: Search query
            top_k: Number of results
            filter_metadata: Metadata filters (applied post-retrieval)
            
        Returns:
            List of SearchResult objects
        """
        if not self.bm25:
            logger.warning("BM25 index not initialized")
            return []
        
        try:
            # Tokenize query
            tokenized_query = query.lower().split()
            
            # Get scores
            scores = self.bm25.get_scores(tokenized_query)
            
            # Create results with indices
            results_with_scores = []
            for idx, score in enumerate(scores):
                if score > 0:  # Only keep relevant results
                    results_with_scores.append((idx, score))
            
            # Sort by score descending
            results_with_scores.sort(key=lambda x: x[1], reverse=True)
            
            # Convert to SearchResults
            search_results = []
            count = 0
            
            for idx, score in results_with_scores:
                if count >= top_k:
                    break
                
                doc = self.documents[idx]
                
                # Apply metadata filter if present
                if filter_metadata:
                    match = True
                    for k, v in filter_metadata.items():
                        if doc.metadata.get(k) != v:
                            match = False
                            break
                    if not match:
                        continue
                
                # Normalize score (BM25 scores are unbounded, but usually < 20-30)
                # Simple normalization for hybrid fusion later
                # This is not perfect but sufficient for RRF
                normalized_score = min(score / 10.0, 1.0)
                
                search_results.append(SearchResult(
                    document=doc,
                    score=normalized_score,
                    id=f"bm25_{idx}"  # Temporary ID
                ))
                count += 1
            
            logger.debug(f"BM25 found {len(search_results)} results")
            return search_results
            
        except Exception as e:
            logger.error(f"BM25 retrieval failed: {e}")
            raise RetrievalError(f"BM25 search failed: {e}")
    
    def _save_index(self):
        """Save index to disk."""
        if not self.persist_dir:
            return
            
        try:
            with open(self.index_path, 'wb') as f:
                pickle.dump({
                    'bm25': self.bm25,
                    'documents': self.documents
                }, f)
            logger.info(f"Saved BM25 index to {self.index_path}")
        except Exception as e:
            logger.error(f"Failed to save BM25 index: {e}")
    
    def _load_index(self):
        """Load index from disk."""
        if not self.index_path.exists():
            return
            
        try:
            with open(self.index_path, 'rb') as f:
                data = pickle.load(f)
                self.bm25 = data['bm25']
                self.documents = data['documents']
            logger.info(f"Loaded BM25 index from {self.index_path}")
        except Exception as e:
            logger.error(f"Failed to load BM25 index: {e}")
