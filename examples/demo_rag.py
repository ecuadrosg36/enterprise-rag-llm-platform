"""
RAG Demo Script.

Demonstrates how to use the RAG pipeline programmatically.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.core.config import get_config
from src.api.dependencies import get_rag_generator

def main():
    print("🤖 Initializing RAG System...")
    
    # Use dependency injection helper to get fully configured generator
    # This automatically handles config, embeddings, vector store, etc.
    rag = get_rag_generator()
    
    print("✅ System Ready!\n")
    
    while True:
        query = input("Enter your question (or 'q' to quit): ").strip()
        if query.lower() == 'q':
            break
            
        if not query:
            continue
            
        print("\n🔍 Thinking...")
        
        try:
            result = rag.generate(query, top_k=3)
            
            print(f"\n📝 Answer:\n{result['answer']}\n")
            
            print("📚 Sources:")
            for i, doc in enumerate(result['source_documents'], 1):
                source = doc['metadata'].get('source', 'Unknown')
                page = doc['metadata'].get('page', 'N/A')
                score = doc.get('score', 0.0)
                print(f"  {i}. {source} (Page {page}) - Score: {score:.3f}")
                
        except Exception as e:
            print(f"❌ Error: {e}")
            
        print("\n" + "-"*50 + "\n")

if __name__ == "__main__":
    main()
