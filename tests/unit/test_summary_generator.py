"""
Unit tests for SummaryGenerator hierarchical summarization logic.
"""

import unittest
from unittest.mock import MagicMock, patch, AsyncMock
from src.llm.generator import SummaryGenerator

class TestSummaryGenerator(unittest.IsolatedAsyncioTestCase):

    def setUp(self):
        with patch("src.llm.generator.tiktoken.get_encoding"), \
             patch("src.llm.generator.AsyncOpenAI"):
            self.generator = SummaryGenerator(doc_type="book")
            self.generator.client = AsyncMock()

    async def test_summarize_hierarchy_single_chunk(self):
        """Test summarization with a single chunk (simple case)."""
        chunks = [{"text": "This is a single chunk of text.", "id": "slug_01_001_hash"}]
        
        # Mock LLM response format: resp.choices[0].message.content
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Chapter Summary"
        
        mock_doc_response = MagicMock()
        mock_doc_response.choices = [MagicMock()]
        mock_doc_response.choices[0].message.content = "Document Summary"
        
        self.generator.client.chat.completions.create.side_effect = [
            mock_response,
            mock_doc_response
        ]
        
        chapters, doc_summary = await self.generator.summarize_hierarchy(chunks)
        
        self.assertEqual(len(chapters), 1)
        self.assertEqual(chapters[0]["summary"], "Chapter Summary")
        self.assertEqual(doc_summary, "Document Summary")

    async def test_summarize_hierarchy_grouping(self):
        """Test that chunks are grouped correctly into chapters."""
        # Create 15 chunks across 2 sections
        chunks = [{"text": f"chunk {i}", "id": f"slug_{1 if i < 10 else 2:02d}_{i:03d}_hash"} for i in range(15)]
        
        # Expect 2 chapter summaries and 1 document summary
        mock_resp = MagicMock()
        mock_resp.choices = [MagicMock()]
        mock_resp.choices[0].message.content = "Summary"
        
        self.generator.client.chat.completions.create.return_value = mock_resp
        
        chapters, doc_summary = await self.generator.summarize_hierarchy(chunks)
        
        self.assertEqual(len(chapters), 2)
        self.assertEqual(doc_summary, "Summary")

    async def test_empty_chunks_handling(self):
        """Test behavior with empty input."""
        # Mock for summarize_book which gets called even with empty chapters if not guarded
        # Wait, if chapters is empty, task is empty, book_summary gets called with []
        mock_resp = MagicMock()
        mock_resp.choices = [MagicMock()]
        mock_resp.choices[0].message.content = ""
        self.generator.client.chat.completions.create.return_value = mock_resp

        chapters, doc_summary = await self.generator.summarize_hierarchy([])
        self.assertEqual(chapters, [])
        self.assertEqual(doc_summary, "")

if __name__ == "__main__":
    unittest.main()
