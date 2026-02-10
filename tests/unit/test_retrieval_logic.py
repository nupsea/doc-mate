"""
Unit tests for AdaptiveRetriever fusion logic and retrieval scenarios.
Uses mocks to simulate BM25, Vector, and Graph retrieval signals.
"""

import unittest
from unittest.mock import MagicMock, patch
import numpy as np
from src.search.adaptive import AdaptiveRetriever

class TestRetrievalLogic(unittest.TestCase):

    def setUp(self):
        # Patching dependencies before init to avoid loading models
        with patch("src.search.vec.SentenceTransformer"), \
             patch("src.search.vec.QdrantClient"), \
             patch("src.search.bm25.PgresStore"):
            self.retriever = AdaptiveRetriever(alpha=0.7)
            # Mock the sub-retrievers
            self.retriever.vec = MagicMock()
            self.retriever.bm25 = MagicMock()

    @patch("src.graph.retriever.PostgresGraphStore")
    @patch("sentence_transformers.SentenceTransformer")
    def test_semantic_entity_matching(self, MockST, MockStore):
        """Test that entities are matched by description semantics."""
        from src.graph.retriever import GraphRetriever
        
        mock_store = MockStore.return_value
        mock_embedder = MockST.return_value
        
        # Setup entities in DB: one matches name, one matches desc semantically
        mock_store.execute.return_value = [
            ("Achilles", 1, "A great Greek warrior"),
            ("King of Ithaca", 2, "The clever hero who wandered for ten years")
        ]
        
        # Mock embeddings
        # query "clever hero" should match entity 2
        mock_embedder.encode.side_effect = lambda texts, **kwargs: \
            np.array([[0.9, 0.1]]) if texts == ["clever hero"] else \
            np.array([[0.1, 0.9], [0.85, 0.15]]) # Embeddings for descriptions
            
        retriever = GraphRetriever(embedder=mock_embedder)
        found_ids = retriever._extract_query_entities("clever hero", 1)
        
        # Should find entity 2 via semantic match of description
        self.assertIn(2, found_ids)
        self.assertNotIn(1, found_ids)

    def test_preprocessing_stop_words(self):
        """Test that common stop words are correctly removed."""
        query = "What is the role of Telemachus in the Odyssey?"
        processed = self.retriever.preprocess_query(query)
        
        self.assertNotIn("what", processed.lower())
        self.assertNotIn("the", processed.lower())
        self.assertIn("telemachus", processed.lower())
        self.assertIn("odyssey", processed.lower())

    def test_fusion_bm25_dominance(self):
        """Test fusion when BM25 has very relevant results and vector doesn't."""
        self.retriever.bm25.search.return_value = [{"id": "chunk_1", "score": 10.0}]
        self.retriever.vec.search.return_value = [{"id": "chunk_99", "score": 0.9}]
        
        results = self.retriever.search("test query", topk=1)
        self.assertEqual(results[0], "chunk_1")

    @patch("src.graph.retriever.ConversationGraphRetriever")
    def test_triple_hybrid_fusion(self, MockGraph):
        """Test that graph results are correctly incorporated into Triple Hybrid Fusion."""
        mock_graph_inst = MockGraph.return_value
        
        self.retriever.bm25.search.return_value = [{"id": "chunk_b"}]
        self.retriever.vec.search.return_value = [{"id": "chunk_v"}]
        mock_graph_inst.search.return_value = [{"id": "chunk_g"}]
        
        results = self.retriever.search("query", doc_slug="test_doc", topk=3)
        
        self.assertIn("chunk_b", results)
        self.assertIn("chunk_v", results)
        self.assertIn("chunk_g", results)
        self.assertTrue(results.index("chunk_g") > results.index("chunk_b"))

if __name__ == "__main__":
    unittest.main()
