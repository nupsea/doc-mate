"""
Unit tests for RRG components (Graph Retriever, Aggregator) logic.
Isolates logic from DB state using mocks.
"""

import unittest
from unittest.mock import MagicMock, patch
from src.flows.agent_graph import graph_retriever_node, context_aggregator_node, AgentState

class TestRRGComponents(unittest.IsolatedAsyncioTestCase):

    @patch("src.graph.store.PostgresGraphStore")
    async def test_graph_retriever_logic(self, MockStore):
        """Test that graph retriever correctly calls DB and formats relationships."""
        mock_store = MockStore.return_value
        
        state = AgentState(
            messages=[MagicMock(content="How is Hector related to Priam?")],
            provider="openai", model="gpt-4", selected_doc_slug="",
            intent="explore", strategy="hybrid",
            target_slugs=["ili"], target_doc_types={"ili": "book"},
            entities=["Hector", "Priam"], router_entities=["Hector", "Priam"],
            partial_summaries={}, partial_passages={}, partial_relations={}, retrieved_context=""
        )

        mock_store._resolve_doc_id.return_value = 1
        mock_store.find_entities_by_names.return_value = {"Hector": 101, "Priam": 102}
        
        mock_store.find_related_entities.side_effect = lambda eid, hops: [
            {"relation_type": "son_of", "name": "Priam", "entity_type": "Person", "depth": 1}
        ] if eid == 101 else [
            {"relation_type": "father_of", "name": "Hector", "entity_type": "Person", "depth": 1}
        ]

        result = await graph_retriever_node(state)
        relations = result["partial_relations"].get("ili", [])
        self.assertIn("Hector son_of Priam (Person)", relations)
        self.assertIn("Priam father_of Hector (Person)", relations)

    def test_aggregator_logic(self):
        """Test that aggregator combines all partial results correctly."""
        state = AgentState(
            target_doc_types={"test_doc": "book"},
            partial_summaries={"test_doc": "This is a summary."},
            partial_passages={"test_doc": [{"text": "Passage 1", "id": "chk1", "metadata": {}}]},
            partial_relations={"test_doc": ["A related_to B"]},
            messages=[MagicMock(content="test query")], provider="", model="", selected_doc_slug="",
            intent="", strategy="", target_slugs=[], entities=[], router_entities=[],
            retrieved_context=""
        )
        
        result = context_aggregator_node(state)
        context = result["retrieved_context"]
        
        self.assertIn("=== DOCUMENT: test_doc (Type: book) ===", context)
        self.assertIn("[OVERVIEW]\nThis is a summary.", context)
        self.assertIn("[KNOWN RELATIONSHIPS]\nA related_to B", context)
        self.assertIn("[RELEVANT PASSAGES]\n1. Passage 1", context)

if __name__ == "__main__":
    unittest.main()
