"""
Migration script to populate PostgreSQL BM25 index from Qdrant data.
This eliminates the need for the legacy pickle-based BM25 index.
"""

import sys
import os
import logging
from tqdm import tqdm

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.search.vec import SemanticRetriever
from src.search.bm25 import BM25Retriever

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def migrate_bm25():
    logger.info("Starting BM25 migration from Qdrant to PostgreSQL...")
    
    vec_retriever = SemanticRetriever()
    bm25_retriever = BM25Retriever()
    
    batch_size = 1000
    chunks_buffer = []
    total_processed = 0
    
    logger.info("Fetching chunks from Qdrant...")
    for chunk in tqdm(vec_retriever.get_all_chunks(batch_size=100)):
        chunks_buffer.append(chunk)
        
        if len(chunks_buffer) >= batch_size:
            logger.info(f"Indexing batch of {len(chunks_buffer)} chunks...")
            bm25_retriever.build_index(chunks_buffer)
            total_processed += len(chunks_buffer)
            chunks_buffer = []
            
    # Process remaining chunks
    if chunks_buffer:
        logger.info(f"Indexing final batch of {len(chunks_buffer)} chunks...")
        bm25_retriever.build_index(chunks_buffer)
        total_processed += len(chunks_buffer)
        
    logger.info(f"Migration complete! Indexed {total_processed} chunks in PostgreSQL.")

if __name__ == "__main__":
    migrate_bm25()
