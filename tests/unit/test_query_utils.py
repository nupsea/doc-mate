"""
Unit tests for query utilities (diversification, timestamp parsing, adjacent chunks).
"""

import unittest
from datetime import datetime
from unittest.mock import patch
from src.flows.document_query import _diversify_conversation_results, _parse_timestamp, get_adjacent_chunks

class TestQueryUtils(unittest.TestCase):

    def test_parse_timestamp(self):
        """Test timestamp parsing with various formats."""
        self.assertEqual(_parse_timestamp("2024-05-12 09:30:00"), datetime(2024, 5, 12, 9, 30, 0))
        self.assertEqual(_parse_timestamp("2024-05-12 09:30"), datetime(2024, 5, 12, 9, 30, 0))
        self.assertEqual(_parse_timestamp("2024-05-12"), datetime(2024, 5, 12, 0, 0, 0))
        self.assertIsNone(_parse_timestamp("invalid-date"))

    def test_diversify_conversation_results(self):
        """Test that conversation results are diversified by speaker and time."""
        chunks = [
            {
                "id": f"chk_{i}",
                "text": f"message {i}",
                "metadata": {"speaker": "Alice", "timestamp": f"2024-05-12 09:{i*10:02d}:00"}
            }
            for i in range(10)
        ]
        
        diversified = _diversify_conversation_results(chunks, target_count=5)
        
        alice_count = sum(1 for c in diversified if c["metadata"]["speaker"] == "Alice")
        self.assertEqual(alice_count, 2)
        self.assertEqual(len(diversified), 2)

    @patch("src.flows.document_query.PgresStore")
    @patch("src.flows.document_query.get_retriever")
    def test_get_adjacent_chunks(self, mock_get_retriever, MockStore):
        """Test fetching adjacent chunks based on ID sequence."""
        mock_store = MockStore.return_value
        mock_retriever = mock_get_retriever.return_value
        
        doc_ids = [
            "conv_01_001_abc",
            "conv_01_002_def",
            "conv_01_003_ghi",
            "conv_01_004_jkl"
        ]
        mock_store.execute.return_value = [(id,) for id in doc_ids]
        
        mock_retriever.vec.get_chunks_by_ids.return_value = [
            {"id": "conv_01_001_abc", "text": "msg 1"},
            {"id": "conv_01_003_ghi", "text": "msg 3"}
        ]
        
        results = get_adjacent_chunks("conv", "conv_01_002_def", before=1, after=1)
        
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0]["id"], "conv_01_001_abc")
        self.assertEqual(results[1]["id"], "conv_01_003_ghi")

if __name__ == "__main__":
    unittest.main()
