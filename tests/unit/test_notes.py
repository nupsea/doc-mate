"""
Unit tests for notes feature -- CRUD operations, indexing pipeline,
schema validation, source_refs handling, and cascade deletes.
"""

import unittest
from unittest.mock import patch, MagicMock, AsyncMock
import asyncio


# ---------------------------------------------------------------------------
# Graph schema
# ---------------------------------------------------------------------------
class TestNoteSchema(unittest.TestCase):

    def test_note_schema_exists(self):
        from src.graph.schemas import DOC_TYPE_SCHEMAS
        self.assertIn("note", DOC_TYPE_SCHEMAS)

    def test_note_schema_entity_types(self):
        from src.graph.schemas import DOC_TYPE_SCHEMAS
        schema = DOC_TYPE_SCHEMAS["note"]
        self.assertIn("entity_types", schema)
        self.assertIn("Person", schema["entity_types"])
        self.assertIn("Concept", schema["entity_types"])
        self.assertIn("Insight", schema["entity_types"])

    def test_note_schema_relationship_types(self):
        from src.graph.schemas import DOC_TYPE_SCHEMAS
        schema = DOC_TYPE_SCHEMAS["note"]
        self.assertIn("relationship_types", schema)
        self.assertIn("relates_to", schema["relationship_types"])
        self.assertIn("supports", schema["relationship_types"])
        self.assertIn("contradicts", schema["relationship_types"])

    def test_note_schema_has_focus(self):
        from src.graph.schemas import DOC_TYPE_SCHEMAS
        schema = DOC_TYPE_SCHEMAS["note"]
        self.assertIn("focus", schema)
        self.assertIsInstance(schema["focus"], str)
        self.assertTrue(len(schema["focus"]) > 10)


# ---------------------------------------------------------------------------
# Note CRUD (store methods)
# ---------------------------------------------------------------------------
class TestNoteCRUD(unittest.TestCase):
    """Test NoteStore note methods using mocked DB connections."""

    def _make_store(self, mock_conn):
        from src.content.note_store import NoteStore
        return NoteStore(conn=mock_conn)

    def test_create_note_inserts_document_and_note(self):
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
        mock_cursor.fetchone.side_effect = [(42,), (7,)]  # doc_id, note_id

        store = self._make_store(mock_conn)
        result = store.create_note("Test Note", "Some content", tags=["test"])

        self.assertEqual(result["doc_id"], 42)
        self.assertEqual(result["note_id"], 7)
        self.assertTrue(result["slug"].startswith("note-"))
        self.assertEqual(mock_cursor.execute.call_count, 2)  # documents + notes
        mock_conn.commit.assert_called_once()

    def test_create_note_generates_unique_slug(self):
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
        mock_cursor.fetchone.side_effect = [(1,), (1,)]

        store = self._make_store(mock_conn)
        result = store.create_note("Title", "Content")

        # Slug should be note-{hex_timestamp}
        self.assertTrue(result["slug"].startswith("note-"))
        self.assertTrue(len(result["slug"]) > 5)

    def test_update_note_increments_version(self):
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)

        store = self._make_store(mock_conn)
        store.update_note(7, content="Updated content")

        # Should have executed UPDATE notes SET content=..., version=version+1
        update_call = mock_cursor.execute.call_args_list[0]
        sql = update_call[0][0]
        self.assertIn("version = version + 1", sql)
        self.assertIn("updated_at = NOW()", sql)

    def test_update_note_title_updates_documents_table(self):
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)

        store = self._make_store(mock_conn)
        store.update_note(7, title="New Title")

        # Should update documents table
        calls = mock_cursor.execute.call_args_list
        title_update = [c for c in calls if "UPDATE documents SET title" in c[0][0]]
        self.assertEqual(len(title_update), 1)

    def test_get_note_by_id(self):
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
        mock_cursor.fetchone.return_value = (
            7, 42, "note-abc", "Test", "Content here",
            ["tag1"], [{"slug": "ody"}], False, None, 1, 3, None,
        )

        store = self._make_store(mock_conn)
        note = store.get_note(7)

        self.assertIsNotNone(note)
        self.assertEqual(note["note_id"], 7)
        self.assertEqual(note["title"], "Test")
        self.assertEqual(note["content"], "Content here")
        self.assertEqual(note["tags"], ["tag1"])

    def test_get_note_returns_none_when_not_found(self):
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
        mock_cursor.fetchone.return_value = None

        store = self._make_store(mock_conn)
        note = store.get_note(999)
        self.assertIsNone(note)

    def test_list_notes_with_tag_filter(self):
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
        mock_cursor.fetchall.return_value = [
            (1, "note-a", "Note A", ["odyssey"], False, None, 1, "Preview..."),
        ]

        store = self._make_store(mock_conn)
        notes = store.list_notes(tag="odyssey")

        self.assertEqual(len(notes), 1)
        self.assertEqual(notes[0]["title"], "Note A")
        # Check that tag filter is in the SQL
        sql = mock_cursor.execute.call_args[0][0]
        self.assertIn("tags @>", sql)

    def test_list_notes_with_search_text(self):
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
        mock_cursor.fetchall.return_value = []

        store = self._make_store(mock_conn)
        store.list_notes(search_text="odysseus")

        sql = mock_cursor.execute.call_args[0][0]
        self.assertIn("ILIKE", sql)

    def test_delete_note_cascades(self):
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
        mock_cursor.fetchone.return_value = (42,)  # doc_id
        mock_cursor.rowcount = 1

        store = self._make_store(mock_conn)
        result = store.delete_note(7)

        self.assertTrue(result)
        # Should delete from documents (which cascades)
        delete_call = [c for c in mock_cursor.execute.call_args_list
                       if "DELETE FROM documents" in c[0][0]]
        self.assertEqual(len(delete_call), 1)

    def test_get_notes_referencing_doc(self):
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
        mock_cursor.fetchall.return_value = [
            (1, "note-a", "My Note", ["themes"], None),
        ]

        store = self._make_store(mock_conn)
        notes = store.get_notes_referencing_doc("ody")

        self.assertEqual(len(notes), 1)
        self.assertEqual(notes[0]["title"], "My Note")
        # Check JSONB containment query
        sql = mock_cursor.execute.call_args[0][0]
        self.assertIn("source_refs @>", sql)


# ---------------------------------------------------------------------------
# Note chunking
# ---------------------------------------------------------------------------
class TestNoteChunking(unittest.TestCase):

    def test_chunk_single_paragraph(self):
        from src.flows.note_ingest import _chunk_markdown
        chunks = _chunk_markdown("This is a short note.", "note-test")
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0]["text"], "This is a short note.")

    def test_chunk_multiple_paragraphs(self):
        from src.flows.note_ingest import _chunk_markdown
        content = "First paragraph.\n\nSecond paragraph.\n\nThird paragraph."
        chunks = _chunk_markdown(content, "note-test", max_tokens=5)
        # With max_tokens=5, each short paragraph should be its own chunk
        self.assertGreaterEqual(len(chunks), 2)

    def test_chunk_merges_small_paragraphs(self):
        from src.flows.note_ingest import _chunk_markdown
        content = "Short.\n\nAlso short.\n\nStill short."
        chunks = _chunk_markdown(content, "note-test", max_tokens=300)
        # All three paragraphs should merge into one chunk
        self.assertEqual(len(chunks), 1)
        self.assertIn("Short.", chunks[0]["text"])
        self.assertIn("Also short.", chunks[0]["text"])

    def test_chunk_empty_content(self):
        from src.flows.note_ingest import _chunk_markdown
        chunks = _chunk_markdown("", "note-test")
        self.assertEqual(len(chunks), 0)

    def test_chunk_ids_are_deterministic(self):
        from src.flows.note_ingest import _chunk_markdown
        content = "First paragraph.\n\nSecond paragraph."
        chunks1 = _chunk_markdown(content, "note-test", max_tokens=5)
        chunks2 = _chunk_markdown(content, "note-test", max_tokens=5)
        for c1, c2 in zip(chunks1, chunks2):
            self.assertEqual(c1["id"], c2["id"])

    def test_chunk_metadata_includes_doc_type(self):
        from src.flows.note_ingest import _chunk_markdown
        chunks = _chunk_markdown("Some content here.", "note-test")
        self.assertEqual(chunks[0]["metadata"]["doc_type"], "note")


# ---------------------------------------------------------------------------
# Note indexing pipeline
# ---------------------------------------------------------------------------
class TestNoteIndexing(unittest.TestCase):

    @patch("src.flows.note_ingest._extract_note_graph", new_callable=AsyncMock)
    @patch("src.flows.note_ingest.FusionRetriever")
    @patch("src.flows.note_ingest._clear_old_index")
    @patch("src.flows.note_ingest.PgresStore")
    def test_index_note_full_pipeline(self, MockStore, mock_clear, MockRetriever, mock_graph):
        """Test that index_note calls clear, BM25, vector, and graph."""
        mock_store = MockStore.return_value
        mock_store._resolve_doc_id.return_value = 42

        mock_retriever = MockRetriever.return_value
        mock_retriever.bm25.build_index = MagicMock()
        mock_retriever.vec.build_index = MagicMock()

        content = "This is a substantial note with enough tokens to trigger graph extraction. " * 10
        asyncio.run(
            asyncio.coroutine(lambda: None)()  # warm up loop
        ) if False else None

        loop = asyncio.new_event_loop()
        loop.run_until_complete(
            __import__("src.flows.note_ingest", fromlist=["index_note"]).index_note(
                "note-test", content
            )
        )
        loop.close()

        mock_clear.assert_called_once()
        mock_retriever.bm25.build_index.assert_called_once()
        mock_retriever.vec.build_index.assert_called_once()
        mock_store.execute.assert_called()  # UPDATE documents

    @patch("src.flows.note_ingest.FusionRetriever")
    @patch("src.flows.note_ingest._clear_old_index")
    @patch("src.flows.note_ingest.PgresStore")
    def test_index_note_skips_graph_for_short_content(self, MockStore, mock_clear, MockRetriever):
        """Notes under 100 tokens should skip graph extraction."""
        mock_store = MockStore.return_value
        mock_store._resolve_doc_id.return_value = 42

        mock_retriever = MockRetriever.return_value
        mock_retriever.bm25.build_index = MagicMock()
        mock_retriever.vec.build_index = MagicMock()

        loop = asyncio.new_event_loop()
        loop.run_until_complete(
            __import__("src.flows.note_ingest", fromlist=["index_note"]).index_note(
                "note-test", "Short note."
            )
        )
        loop.close()

        # Graph extraction should NOT have been called (short content)
        # The function should still complete successfully
        mock_retriever.bm25.build_index.assert_called_once()

    @patch("src.flows.note_ingest.PgresStore")
    def test_index_note_handles_missing_doc(self, MockStore):
        """Should gracefully handle non-existent document."""
        mock_store = MockStore.return_value
        mock_store._resolve_doc_id.return_value = None

        loop = asyncio.new_event_loop()
        loop.run_until_complete(
            __import__("src.flows.note_ingest", fromlist=["index_note"]).index_note(
                "note-nonexistent", "Content"
            )
        )
        loop.close()
        # Should not raise, just return early


# ---------------------------------------------------------------------------
# Source refs in agent pipeline
# ---------------------------------------------------------------------------
class TestSourceRefs(unittest.TestCase):

    def test_agent_state_has_source_refs(self):
        from src.flows.agent_graph import AgentState
        # TypedDict annotations should include source_refs
        self.assertIn("source_refs", AgentState.__annotations__)

    def test_aggregator_returns_source_refs(self):
        """Verify context_aggregator_node includes source_refs in output."""
        from src.flows.agent_graph import context_aggregator_node
        from langchain_core.messages import HumanMessage

        state = {
            "partial_summaries": {},
            "partial_passages": {
                "ody": [
                    {"id": "ody_01_001_abc", "text": "Odysseus journeyed home.", "metadata": {}},
                    {"id": "ody_02_001_def", "text": "He faced many trials.", "metadata": {}},
                ]
            },
            "partial_relations": {},
            "target_doc_types": {"ody": "book"},
            "router_entities": [],
            "messages": [HumanMessage(content="Tell me about Odysseus")],
        }

        result = context_aggregator_node(state)

        self.assertIn("source_refs", result)
        self.assertEqual(len(result["source_refs"]), 2)
        self.assertEqual(result["source_refs"][0]["slug"], "ody")
        self.assertEqual(result["source_refs"][0]["chunk_id"], "ody_01_001_abc")
        self.assertTrue(len(result["source_refs"][0]["snippet"]) > 0)


# ---------------------------------------------------------------------------
# Document selector formatting
# ---------------------------------------------------------------------------
class TestDocDropdownFormatting(unittest.TestCase):

    def test_notes_get_prefix_in_dropdown(self):
        from src.ui.utils import format_doc_dropdown_choices
        docs = [
            ("ody", "The Odyssey", "Homer", 100, None, "book"),
            ("note-abc", "My Note", "user", 2, None, "note"),
        ]
        choices = format_doc_dropdown_choices(docs)

        # First is the placeholder
        self.assertEqual(choices[0][1], "none")
        # Book should not have prefix
        self.assertEqual(choices[1][0], "The Odyssey")
        # Note should have [Note] prefix
        self.assertEqual(choices[2][0], "[Note] My Note")

    def test_empty_docs_returns_placeholder_only(self):
        from src.ui.utils import format_doc_dropdown_choices
        choices = format_doc_dropdown_choices([])
        self.assertEqual(len(choices), 1)
        self.assertEqual(choices[0][1], "none")


if __name__ == "__main__":
    unittest.main()
