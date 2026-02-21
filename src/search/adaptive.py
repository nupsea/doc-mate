"""
Adaptive hybrid search with dynamic alpha and query preprocessing.
"""

import re
import logging
from src.search.hybrid import FusionRetriever

logger = logging.getLogger(__name__)


class AdaptiveRetriever(FusionRetriever):
    """
    Enhanced FusionRetriever with:
    1. Dynamic alpha based on query type
    2. Query preprocessing (stop word removal)
    3. Increased candidate pool for better fusion
    """

    # Common stop words to remove
    STOP_WORDS = {
        "what",
        "when",
        "where",
        "who",
        "why",
        "how",
        "does",
        "do",
        "did",
        "is",
        "are",
        "was",
        "were",
        "the",
        "a",
        "an",
        "about",
        "in",
        "on",
        "at",
        "to",
        "for",
        "of",
        "with",
        "by",
        "from",
        "this",
        "that",
        "these",
        "those",
        "and",
        "or",
        "but",
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.base_alpha = kwargs.get("alpha", 0.7)

    def preprocess_query(self, query: str) -> str:
        """
        Preprocess query to improve keyword matching.

        Args:
            query: Raw user query

        Returns:
            Preprocessed query with stop words removed
        """
        # Convert to lowercase
        query_lower = query.lower()

        # Remove punctuation except hyphens
        query_clean = re.sub(r"[^\w\s-]", " ", query_lower)

        # Split and filter stop words
        words = query_clean.split()
        filtered_words = [w for w in words if w not in self.STOP_WORDS and len(w) > 2]

        # If we filtered out everything, return original query
        if not filtered_words:
            return query

        return " ".join(filtered_words)

    def get_dynamic_alpha(self, query: str) -> float:
        """
        Determine optimal alpha based on query characteristics.

        Note: Testing showed fixed α=0.7 performs better than dynamic alpha.
        This method is kept for potential future use.

        Args:
            query: User query

        Returns:
            Dynamic alpha value (0.0 to 1.0)
        """
        # Question queries benefit from semantic search
        if "?" in query:
            return 0.5  # Balanced approach for questions

        # Very short keyword queries benefit from BM25
        word_count = len(query.split())
        if word_count <= 3:
            return 0.8  # Heavy BM25 for keywords

        # Medium-length queries use base alpha
        if word_count <= 8:
            return self.base_alpha  # Default 0.7

        # Long descriptive queries benefit from semantic
        return 0.6  # Slightly more semantic weight

    def _query_semantic_ratio(self, query: str) -> float:
        """Estimate how much a query benefits from semantic vs keyword matching.

        Returns a value in [0, 1] where higher means more semantic.
        Heuristics:
        - Questions and longer queries lean semantic
        - Short keyword queries lean BM25
        - Quoted phrases lean BM25
        """
        words = query.split()
        word_count = len(words)

        if '"' in query or "'" in query:
            return 0.2  # Quoted phrase -- user wants exact match
        if "?" in query or word_count > 6:
            return 0.7  # Natural language question
        if word_count <= 2:
            return 0.2  # Short keyword lookup
        return 0.4  # Default moderate

    def search(self,
        query: str,
        topk: int = 7,
        use_preprocessing: bool = True,
        use_dynamic_alpha: bool = False,
        candidate_multiplier: int = 3,
        doc_slug: str = None,
        hint_entities: list[str] = None,
        doc_type: str = None,
    ) -> list[str]:
        """
        Adaptive search with query preprocessing.

        Key improvement: Query preprocessing (stop word removal) provides +8% Hit@5.
        Recommended: use_preprocessing=True, use_dynamic_alpha=False (fixed α=0.7)

        Args:
            query: Search query
            topk: Number of results to return
            use_preprocessing: Whether to preprocess query (recommended: True)
            use_dynamic_alpha: Whether to use dynamic alpha (recommended: False, use fixed α=0.7)
            candidate_multiplier: Retrieve topk * multiplier candidates before fusion
            doc_slug: If provided, only search within this document (e.g., 'aiw', 'gtr')
            hint_entities: Optional hint entities from router to seed graph search
            doc_type: Document type (conversation, book, etc.) for adaptive weighting

        Returns:
            List of chunk IDs
        """
        # Determine alpha
        if use_dynamic_alpha:
            alpha = self.get_dynamic_alpha(query)
        else:
            alpha = self.base_alpha

        # Preprocess query
        if use_preprocessing:
            processed_query = self.preprocess_query(query)
        else:
            processed_query = query

        # Conversations benefit from a larger candidate pool
        if doc_type == "conversation":
            candidate_multiplier = max(candidate_multiplier, 5)

        # Retrieve more candidates for better fusion
        candidate_count = topk * candidate_multiplier

        # Get results from both systems with document filtering
        embed_results = self.vec.search(query, candidate_count, doc_slug=doc_slug)
        bm25_results = self.bm25.search(processed_query, candidate_count, doc_slug=doc_slug)

        # Graph retrieval (third signal)
        graph_results = []
        if doc_slug:
            try:
                from src.graph.retriever import ConversationGraphRetriever
                # Use ConversationGraphRetriever as it handles both basic and episode queries
                # (It falls back to basic entity search if no episodes match)
                graph_retriever = ConversationGraphRetriever()
                graph_results = graph_retriever.search(query, doc_slug, candidate_count, hint_entities=hint_entities)
                logger.info(f"Graph retrieval found {len(graph_results)} results for '{query}' in {doc_slug}")
            except Exception as e:
                logger.warning(f"Graph retrieval failed: {e}")

        # Adaptive Fusion Weights
        sem_ratio = self._query_semantic_ratio(query)

        if graph_results:
            # Triple Hybrid -- shift BM25/Vector balance by query semantics
            alpha_graph = 0.30
            remaining = 1.0 - alpha_graph
            alpha_vec = remaining * (0.4 + 0.2 * sem_ratio)   # 0.28 - 0.42
            alpha_bm25 = remaining - alpha_vec                  # 0.42 - 0.28
        elif doc_type == "conversation":
            # Conversations: boost semantic weight (paraphrasing is common)
            alpha_vec = 0.35 + 0.15 * sem_ratio   # 0.35 - 0.50
            alpha_bm25 = 1.0 - alpha_vec           # 0.65 - 0.50
            alpha_graph = 0.0
        else:
            # Dual Hybrid (Classic) with mild adaptive shift
            alpha_vec = (1.0 - alpha) + 0.10 * sem_ratio  # base 0.30 + up to 0.07
            alpha_bm25 = 1.0 - alpha_vec
            alpha_graph = 0.0

        logger.info("Fusion weights: bm25=%.2f, vec=%.2f, graph=%.2f (sem_ratio=%.2f, doc_type=%s)",
                    alpha_bm25, alpha_vec, alpha_graph, sem_ratio, doc_type)

        # Apply weighted fusion (RRF-style reciprocal rank or simple score sum? Using Rank Fusion here)
        scores = {}
        
        # BM25 Scores
        for rank, c in enumerate(bm25_results, start=1):
            scores[c["id"]] = scores.get(c["id"], 0) + alpha_bm25 * (1.0 / rank)

        # Vector Scores
        for rank, c in enumerate(embed_results, start=1):
            scores[c["id"]] = scores.get(c["id"], 0) + alpha_vec * (1.0 / rank)
            
        # Graph Scores
        if graph_results:
            for rank, c in enumerate(graph_results, start=1):
                # Graph results come with their own scores, but for fusion consistency 
                # we treat them by rank here to normalize with others
                scores[c["id"]] = scores.get(c["id"], 0) + alpha_graph * (1.0 / rank)

        # Sort and return top-k
        sorted_results = sorted(scores.items(), key=lambda x: -x[1])[:topk]
        return [cid for cid, _ in sorted_results]

    def id_search(self, query: str, topk: int = 7, **kwargs) -> list[str]:
        """Alias for search() to maintain compatibility."""
        return self.search(query, topk, **kwargs)


if __name__ == "__main__":
    # Test preprocessing
    retriever = AdaptiveRetriever()

    test_queries = [
        "What does Telemachus feel about the suitors?",
        "Odysseus Cyclops",
        "Why did Ulysses reveal his true name to the Cyclops?",
        "golden sandals",
    ]

    print("=== Query Preprocessing Test ===\n")
    for query in test_queries:
        processed = retriever.preprocess_query(query)
        alpha = retriever.get_dynamic_alpha(query)
        print(f"Original: {query}")
        print(f"Processed: {processed}")
        print(f"Alpha: {alpha}")
        print()
