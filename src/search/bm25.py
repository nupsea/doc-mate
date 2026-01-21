import math
import re
import logging
from collections import Counter
from src.content.store import PgresStore

logger = logging.getLogger(__name__)

STOPWORDS = {"the", "a", "an", "and", "of", "in", "to"}


def simple_tokenize(text):
    return [w for w in re.findall(r"\w+", text.lower()) if w not in STOPWORDS]


class BM25Retriever:
    def __init__(self, k1=1.5, b=0.75):
        self.k1 = k1
        self.b = b
        self.store = PgresStore()

    def build_index(self, chunks):
        """
        Build BM25 index from chunks and store in DB.
        """
        term_freqs = []
        doc_lens = []

        for chunk in chunks:
            chunk_id = chunk["id"]
            tokens = simple_tokenize(chunk["text"])
            doc_len = len(tokens)
            doc_lens.append((chunk_id, doc_len))
            
            # Count term frequencies for this chunk
            freqs = Counter(tokens)
            for term, freq in freqs.items():
                term_freqs.append((term, chunk_id, freq))

        # Store in DB
        self.store.store_bm25_index(term_freqs, doc_lens)
        logger.info(f"Built and stored BM25 index for {len(chunks)} chunks")

    def score(self, query_tokens, chunk_id, tf, doc_len, df, N, avgdl):
        score = 0.0
        for term in query_tokens:
            if term not in tf.get(chunk_id, {}):
                continue
            
            term_tf = tf[chunk_id][term]
            term_df = df.get(term, 0)
            
            # Calculate IDF
            idf = math.log((N - term_df + 0.5) / (term_df + 0.5) + 1)
            
            # Calculate BM25 score for this term
            numer = term_tf * (self.k1 + 1)
            denom = term_tf + self.k1 * (1 - self.b + self.b * doc_len / avgdl)
            score += idf * (numer / denom)
            
        return score

    def search(self, query, topk=7, doc_slug=None):
        query_tokens = simple_tokenize(query)
        logger.info("BM25 search: query='%s', topk=%d, doc_slug=%s", query, topk, doc_slug)

        # 1. Fetch Stats from DB
        stats = self.store.get_bm25_stats(query_tokens, doc_slug=doc_slug)
        N = stats["N"]
        if N == 0:
            return []
            
        df = stats["df"]
        tf = stats["tf"]
        doc_lens = stats["doc_lens"]
        
        # Calculate avgdl based on fetched docs (approximate but effective)
        avgdl = sum(doc_lens.values()) / len(doc_lens) if doc_lens else 0

        # 2. Score documents
        scores = []
        for chunk_id, length in doc_lens.items():
            s = self.score(query_tokens, chunk_id, tf, length, df, N, avgdl)
            if s > 0:
                scores.append((chunk_id, s))

        # 3. Rank and Format
        ranked = sorted(scores, key=lambda x: -x[1])[:topk]
        
        # Return results compatible with old interface
        results = [
            {"id": cid, "score": s, "text": "(text fetch required)"} for cid, s in ranked
        ]
        
        logger.info("BM25 returned %d results", len(results))
        return results

    def id_search(self, query: str, topk=7, doc_slug=None):
        search_results = self.search(query, topk, doc_slug)
        return [c["id"] for c in search_results]

    # Deprecated methods stubs for compatibility
    def save_index(self, filepath: str = "bm25_index.pkl"):
        logger.warning("save_index is deprecated. Index is stored in DB.")

    def load_index(self, filepath: str = "bm25_index.pkl"):
        logger.warning("load_index is deprecated. Index is loaded from DB.")

    def cleanup(self):
        logger.warning("cleanup is deprecated.")