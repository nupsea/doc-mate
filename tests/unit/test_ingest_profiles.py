"""
Unit tests for ingestion profiles -- profile definitions, time estimation,
registry, Ollama health check, and integration with document_ingest pipeline.
"""

import asyncio
import unittest
from unittest.mock import patch, MagicMock, AsyncMock

from src.flows.ingest_profiles import (
    IngestProfile,
    PROFILES,
    DEFAULT_PROFILE,
    check_ollama_available,
)


# ---------------------------------------------------------------------------
# Profile registry
# ---------------------------------------------------------------------------
class TestProfileRegistry(unittest.TestCase):

    def test_all_five_profiles_present(self):
        expected = {"openai_full", "openai_fast", "hybrid", "local_full", "search_only"}
        self.assertEqual(set(PROFILES.keys()), expected)

    def test_default_profile_exists(self):
        self.assertIn(DEFAULT_PROFILE, PROFILES)
        self.assertEqual(DEFAULT_PROFILE, "openai_full")

    def test_each_profile_has_required_fields(self):
        required = [
            "name", "label", "description", "summary_provider", "graph_provider",
            "graph_coverage", "graph_sample_pct", "summary_sampling",
            "summary_semaphore", "graph_semaphore", "graph_batch_size",
        ]
        for key, profile in PROFILES.items():
            for field in required:
                self.assertTrue(
                    hasattr(profile, field),
                    f"Profile '{key}' missing field '{field}'",
                )

    def test_provider_values_are_valid(self):
        valid_summary = {"openai", "local"}
        valid_graph = {"openai", "local", "skip"}
        for key, p in PROFILES.items():
            self.assertIn(p.summary_provider, valid_summary, f"{key} summary_provider")
            self.assertIn(p.graph_provider, valid_graph, f"{key} graph_provider")

    def test_graph_coverage_values_are_valid(self):
        valid = {"full", "sampled", "skip"}
        for key, p in PROFILES.items():
            self.assertIn(p.graph_coverage, valid, f"{key} graph_coverage")

    def test_search_only_skips_graph(self):
        p = PROFILES["search_only"]
        self.assertEqual(p.graph_provider, "skip")
        self.assertEqual(p.graph_coverage, "skip")

    def test_openai_full_is_current_default_behavior(self):
        p = PROFILES["openai_full"]
        self.assertEqual(p.summary_provider, "openai")
        self.assertEqual(p.graph_provider, "openai")
        self.assertEqual(p.graph_coverage, "full")
        self.assertEqual(p.graph_sample_pct, 1.0)
        self.assertEqual(p.graph_batch_size, 8)


# ---------------------------------------------------------------------------
# Time estimation
# ---------------------------------------------------------------------------
class TestTimeEstimation(unittest.TestCase):

    def test_estimate_returns_min_max_tuple(self):
        for key, p in PROFILES.items():
            result = p.estimate_seconds(100, "book")
            self.assertIsInstance(result, tuple, f"{key}")
            self.assertEqual(len(result), 2, f"{key}")
            min_s, max_s = result
            self.assertGreater(min_s, 0, f"{key} min should be > 0")
            self.assertGreaterEqual(max_s, min_s, f"{key} max >= min")

    def test_more_chunks_means_longer_estimate(self):
        p = PROFILES["openai_full"]
        _, max_small = p.estimate_seconds(10, "book")
        _, max_large = p.estimate_seconds(500, "book")
        self.assertGreater(max_large, max_small)

    def test_skip_graph_is_fastest(self):
        _, max_skip = PROFILES["search_only"].estimate_seconds(200, "book")
        _, max_full = PROFILES["openai_full"].estimate_seconds(200, "book")
        self.assertLess(max_skip, max_full)

    def test_sampled_graph_faster_than_full(self):
        _, max_fast = PROFILES["openai_fast"].estimate_seconds(200, "conversation")
        _, max_full = PROFILES["openai_full"].estimate_seconds(200, "conversation")
        self.assertLess(max_fast, max_full)

    def test_conversation_type_adds_episode_time(self):
        p = PROFILES["openai_full"]
        _, max_book = p.estimate_seconds(200, "book")
        _, max_conv = p.estimate_seconds(200, "conversation")
        # Conversations have episode extraction on top of entity extraction
        self.assertGreater(max_conv, max_book)

    def test_single_chunk_estimate(self):
        for key, p in PROFILES.items():
            min_s, max_s = p.estimate_seconds(1, "book")
            self.assertGreater(min_s, 0, f"{key}")
            # Single chunk should be fast (< 5 min for any profile)
            self.assertLess(min_s, 300, f"{key}")

    def test_zero_graph_time_when_skipped(self):
        p = PROFILES["search_only"]
        min_s, max_s = p.estimate_seconds(1000, "conversation")
        # Even with 1000 chunks, skip-graph should be short (summary only)
        self.assertLess(max_s, 600)  # Under 10 minutes


# ---------------------------------------------------------------------------
# Ollama availability
# ---------------------------------------------------------------------------
class TestOllamaAvailability(unittest.TestCase):

    @patch("requests.get")
    @patch("src.llm.config.LLMConfig.from_env")
    def test_ollama_available(self, mock_config, mock_get):
        mock_config.return_value.ollama_base_url = "http://localhost:11434"
        mock_get.return_value.status_code = 200
        self.assertTrue(check_ollama_available())
        mock_get.assert_called_once_with("http://localhost:11434/api/tags", timeout=3)

    @patch("requests.get")
    @patch("src.llm.config.LLMConfig.from_env")
    def test_ollama_unavailable_connection_error(self, mock_config, mock_get):
        mock_config.return_value.ollama_base_url = "http://localhost:11434"
        mock_get.side_effect = ConnectionError("refused")
        self.assertFalse(check_ollama_available())

    @patch("requests.get")
    @patch("src.llm.config.LLMConfig.from_env")
    def test_ollama_unavailable_timeout(self, mock_config, mock_get):
        mock_config.return_value.ollama_base_url = "http://localhost:11434"
        mock_get.side_effect = TimeoutError("timed out")
        self.assertFalse(check_ollama_available())

    @patch("requests.get")
    @patch("src.llm.config.LLMConfig.from_env")
    def test_ollama_bad_status_code(self, mock_config, mock_get):
        mock_config.return_value.ollama_base_url = "http://localhost:11434"
        mock_get.return_value.status_code = 500
        self.assertFalse(check_ollama_available())


# ---------------------------------------------------------------------------
# Profile integration with document_ingest pipeline
# ---------------------------------------------------------------------------
class TestProfileIntegration(unittest.IsolatedAsyncioTestCase):
    """Test that profile settings flow correctly through the ingest pipeline."""

    @patch("src.flows.document_ingest.build_graph_index")
    @patch("src.flows.document_ingest.build_search_indexes")
    @patch("src.flows.document_ingest.generate_summaries")
    @patch("src.flows.document_ingest.store_document_metadata")
    @patch("src.flows.document_ingest.get_parser")
    @patch("src.flows.document_ingest.validate_inputs")
    @patch("src.flows.document_ingest.read_and_parse")
    @patch("src.flows.document_ingest.verify_ingestion")
    async def test_profile_settings_passed_to_generate_summaries(
        self, mock_verify, mock_parse, mock_validate, mock_get_parser,
        mock_store_meta, mock_gen_sum, mock_build_search, mock_build_graph,
    ):
        from src.flows.document_ingest import ingest_document

        mock_validate.return_value = {"slug": "test", "file_path": "/tmp/t.txt", "title": "T", "exists": False, "file_size": 100}
        mock_parse.return_value = {"chunks": [{"id": "test_01_001_abc", "text": "hello", "num_tokens": 10}], "num_chunks": 1, "num_chars": 5, "num_tokens": 10}
        mock_store_meta.return_value = {"doc_id": 1, "slug": "test"}
        mock_gen_sum.return_value = {"chapter_summaries": [], "document_summary": "", "num_chapters": 0}
        mock_build_search.return_value = {"bm25_indexed": 1, "vector_indexed": 1, "new_chunks": 1}
        mock_build_graph.return_value = {"entities": 0, "relationships": 0, "episodes": 0, "status": "ok", "summary_text": ""}
        mock_verify.return_value = {"status": "success", "slug": "test", "chapters_verified": 0, "document_summary_length": 0}

        # Test with hybrid profile
        await ingest_document(
            slug="test", file_path="/tmp/t.txt", title="T",
            doc_type="book", profile_name="hybrid",
        )

        # Verify generate_summaries was called with local provider
        call_kwargs = mock_gen_sum.call_args
        self.assertEqual(call_kwargs.kwargs.get("provider") or call_kwargs[1].get("provider"), "local")

    @patch("src.flows.document_ingest.build_graph_index")
    @patch("src.flows.document_ingest.build_search_indexes")
    @patch("src.flows.document_ingest.generate_summaries")
    @patch("src.flows.document_ingest.store_document_metadata")
    @patch("src.flows.document_ingest.get_parser")
    @patch("src.flows.document_ingest.validate_inputs")
    @patch("src.flows.document_ingest.read_and_parse")
    @patch("src.flows.document_ingest.verify_ingestion")
    async def test_search_only_profile_skips_graph(
        self, mock_verify, mock_parse, mock_validate, mock_get_parser,
        mock_store_meta, mock_gen_sum, mock_build_search, mock_build_graph,
    ):
        from src.flows.document_ingest import ingest_document

        mock_validate.return_value = {"slug": "test", "file_path": "/tmp/t.txt", "title": "T", "exists": False, "file_size": 100}
        mock_parse.return_value = {"chunks": [{"id": "test_01_001_abc", "text": "hello", "num_tokens": 10}], "num_chunks": 1, "num_chars": 5, "num_tokens": 10}
        mock_store_meta.return_value = {"doc_id": 1, "slug": "test"}
        mock_gen_sum.return_value = {"chapter_summaries": [], "document_summary": "", "num_chapters": 0}
        mock_build_search.return_value = {"bm25_indexed": 1, "vector_indexed": 1, "new_chunks": 1}
        mock_build_graph.return_value = {"entities": 0, "relationships": 0, "episodes": 0, "status": "skipped", "summary_text": ""}
        mock_verify.return_value = {"status": "success", "slug": "test", "chapters_verified": 0, "document_summary_length": 0}

        await ingest_document(
            slug="test", file_path="/tmp/t.txt", title="T",
            doc_type="book", profile_name="search_only",
        )

        # Verify build_graph_index was called with skip settings
        call_kwargs = mock_build_graph.call_args
        self.assertEqual(call_kwargs.kwargs.get("graph_provider") or call_kwargs[1].get("graph_provider"), "skip")
        self.assertEqual(call_kwargs.kwargs.get("graph_coverage") or call_kwargs[1].get("graph_coverage"), "skip")

    @patch("src.flows.document_ingest.build_graph_index")
    @patch("src.flows.document_ingest.build_search_indexes")
    @patch("src.flows.document_ingest.generate_summaries")
    @patch("src.flows.document_ingest.store_document_metadata")
    @patch("src.flows.document_ingest.get_parser")
    @patch("src.flows.document_ingest.validate_inputs")
    @patch("src.flows.document_ingest.read_and_parse")
    @patch("src.flows.document_ingest.verify_ingestion")
    async def test_default_profile_when_none_specified(
        self, mock_verify, mock_parse, mock_validate, mock_get_parser,
        mock_store_meta, mock_gen_sum, mock_build_search, mock_build_graph,
    ):
        from src.flows.document_ingest import ingest_document

        mock_validate.return_value = {"slug": "test", "file_path": "/tmp/t.txt", "title": "T", "exists": False, "file_size": 100}
        mock_parse.return_value = {"chunks": [{"id": "test_01_001_abc", "text": "hello", "num_tokens": 10}], "num_chunks": 1, "num_chars": 5, "num_tokens": 10}
        mock_store_meta.return_value = {"doc_id": 1, "slug": "test"}
        mock_gen_sum.return_value = {"chapter_summaries": [], "document_summary": "", "num_chapters": 0}
        mock_build_search.return_value = {"bm25_indexed": 1, "vector_indexed": 1, "new_chunks": 1}
        mock_build_graph.return_value = {"entities": 0, "relationships": 0, "episodes": 0, "status": "ok", "summary_text": ""}
        mock_verify.return_value = {"status": "success", "slug": "test", "chapters_verified": 0, "document_summary_length": 0}

        # No profile_name -> should use openai_full defaults
        await ingest_document(
            slug="test", file_path="/tmp/t.txt", title="T", doc_type="book",
        )

        # Verify generate_summaries was called with openai provider
        call_kwargs = mock_gen_sum.call_args
        self.assertEqual(call_kwargs.kwargs.get("provider") or call_kwargs[1].get("provider"), "openai")

        # Verify build_graph_index was called with full coverage
        call_kwargs = mock_build_graph.call_args
        self.assertEqual(call_kwargs.kwargs.get("graph_provider") or call_kwargs[1].get("graph_provider"), "openai")
        self.assertEqual(call_kwargs.kwargs.get("graph_coverage") or call_kwargs[1].get("graph_coverage"), "full")


# ---------------------------------------------------------------------------
# Graph sampling in build_graph_index
# ---------------------------------------------------------------------------
class TestGraphSampling(unittest.IsolatedAsyncioTestCase):

    @patch("src.flows.document_ingest.PostgresGraphStore")
    @patch("src.flows.document_ingest.EntityResolver")
    @patch("src.flows.document_ingest.EntityExtractor")
    async def test_graph_skip_returns_immediately(self, MockExtractor, MockResolver, MockStore):
        from src.flows.document_ingest import build_graph_index

        chunks = [{"id": f"t_{i:02d}_001_abc", "text": f"chunk {i}"} for i in range(100)]
        result = await build_graph_index(
            chunks, doc_id=1, doc_type="book",
            graph_provider="skip", graph_coverage="skip",
        )

        self.assertEqual(result["entities"], 0)
        self.assertEqual(result["status"], "skipped")
        # EntityExtractor should never be instantiated
        MockExtractor.assert_not_called()

    @patch("src.flows.document_ingest._extract_entities_batch", new_callable=AsyncMock)
    @patch("src.flows.document_ingest.PostgresGraphStore")
    @patch("src.flows.document_ingest.EntityResolver")
    @patch("src.flows.document_ingest.EntityExtractor")
    async def test_graph_sampling_reduces_chunks(self, MockExtractor, MockResolver, MockStore, mock_extract):
        from src.flows.document_ingest import build_graph_index

        mock_extract.return_value = (0, 0)
        MockStore.return_value.delete_graph_for_doc = MagicMock()

        chunks = [{"id": f"t_{i:02d}_001_abc", "text": f"chunk {i}"} for i in range(100)]
        await build_graph_index(
            chunks, doc_id=1, doc_type="book",
            graph_provider="openai", graph_coverage="sampled", graph_sample_pct=0.5,
        )

        # _extract_entities_batch should have been called with ~50 chunks, not 100
        call_args = mock_extract.call_args
        passed_chunks = call_args[0][4]  # 5th positional arg is chunks
        self.assertAlmostEqual(len(passed_chunks), 50, delta=5)


# ---------------------------------------------------------------------------
# Summary sampling modes in generate_summaries
# ---------------------------------------------------------------------------
class TestSummarySampling(unittest.IsolatedAsyncioTestCase):

    @patch("src.flows.document_ingest.store_summaries_to_db")
    @patch("src.flows.document_ingest.SummaryGenerator")
    @patch("src.monitoring.tracer.is_phoenix_enabled", return_value=False)
    async def test_aggressive_sampling_threshold(self, mock_phoenix, MockGen, mock_store_db):
        from src.flows.document_ingest import generate_summaries

        mock_gen = MockGen.return_value
        mock_gen.summarize_hierarchy = AsyncMock(return_value=([], ""))

        # 60 conversation chunks -- below normal threshold (100) but above aggressive (50)
        chunks = [{"id": f"t_{i:02d}_001_abc", "text": f"chunk {i}"} for i in range(60)]
        await generate_summaries(
            chunks, doc_type="conversation", summary_sampling="aggressive",
        )

        # With aggressive sampling, 60 chunks should be sampled down to 20
        call_args = mock_gen.summarize_hierarchy.call_args[0]
        self.assertEqual(len(call_args[0]), 20)

    @patch("src.flows.document_ingest.store_summaries_to_db")
    @patch("src.flows.document_ingest.SummaryGenerator")
    @patch("src.monitoring.tracer.is_phoenix_enabled", return_value=False)
    async def test_auto_sampling_uses_normal_threshold(self, mock_phoenix, MockGen, mock_store_db):
        from src.flows.document_ingest import generate_summaries

        mock_gen = MockGen.return_value
        mock_gen.summarize_hierarchy = AsyncMock(return_value=([], ""))

        # 60 conversation chunks -- below auto threshold (100), should NOT be sampled
        chunks = [{"id": f"t_{i:02d}_001_abc", "text": f"chunk {i}"} for i in range(60)]
        await generate_summaries(
            chunks, doc_type="conversation", summary_sampling="auto",
        )

        call_args = mock_gen.summarize_hierarchy.call_args[0]
        self.assertEqual(len(call_args[0]), 60)  # All chunks passed through

    @patch("src.flows.document_ingest.store_summaries_to_db")
    @patch("src.flows.document_ingest.SummaryGenerator")
    @patch("src.monitoring.tracer.is_phoenix_enabled", return_value=False)
    async def test_provider_passed_to_summary_generator(self, mock_phoenix, MockGen, mock_store_db):
        from src.flows.document_ingest import generate_summaries

        mock_gen = MockGen.return_value
        mock_gen.summarize_hierarchy = AsyncMock(return_value=([], ""))

        chunks = [{"id": "t_01_001_abc", "text": "hello"}]
        await generate_summaries(chunks, doc_type="book", provider="local")

        MockGen.assert_called_once_with(doc_type="book", provider="local")


if __name__ == "__main__":
    unittest.main()
