"""
Router logic for determining intent and target documents.
"""

import json
import os
from typing import Dict, Any
from langchain_core.messages import SystemMessage, HumanMessage
from src.content.store import PgresStore

class QueryRouter:
    """
    Analyzes user query to determine:
    1. Intent (search, compare, explore, chat)
    2. Target Slugs (list of document identifiers)
    """
    
    def __init__(self, provider: str = "openai", model: str = "gpt-4o-mini"):
        self.provider = provider
        self.model = model

    def _get_available_docs_context(self) -> str:
        """Fetch available documents for the prompt context."""
        try:
            store = PgresStore()
            with store.conn.cursor() as cur:
                cur.execute("SELECT slug, title, author, doc_type FROM documents")
                docs = cur.fetchall()
            if not docs:
                return "No documents available."
            # Format: - [slug] Title (by Author) [type]
            return "\n".join([f"- [{slug}] {title} (by {author}) [{dtype}]" for slug, title, author, dtype in docs])
        except Exception as e:
            return f"Error fetching docs: {e}"

    def route(self, query: str, selected_slug: str = None) -> Dict[str, Any]:
        """Classifies the query and identifies target documents."""
        doc_list = self._get_available_docs_context()
        
        system_prompt = f"""You are a query router for a document assistant.
        
AVAILABLE DOCUMENTS:
{doc_list}

INTENTS:
- "search": Single document lookup.
- "compare": Multi-document comparison or contrast.
- "explore": Questions about relationships/entities (Knowledge Graph).
- "chat": General conversation not requiring document context.

STRATEGIES:
- "summary": Use ONLY for very broad questions like "What is this about?" or "Summarize themes".
- "search": Use for specific facts, quotes, or narrow details within a text.
- "hybrid": Use for comparisons, relationships, connections, or any "explore" intent where both overview and specific details are needed.

TASK:
Identify the intent, the retrieval strategy, and the relevant document slugs from the query.
For questions about people (Who is X?), relationships (How is X related to Y?), or comparisons (Compare X and Y), ALWAYS use the "hybrid" strategy.

OUTPUT RULES:
- Return ONLY valid JSON.
- Format: {{ "intent": "string", "strategy": "string", "slugs": ["list", "of", "slugs"], "entities": ["list", "of", "people/places/topics"] }}
- Map names and terms mentioned in the query to the [slug] identifiers in the list above.
- "entities": Identify all individuals, locations, or key concepts mentioned in the query to facilitate Knowledge Graph exploration.
- If a slug is explicitly provided in the UI selection below, prioritize it.

UI SELECTION: {selected_slug if selected_slug and selected_slug != 'none' else 'None'}
"""
        response_text = self._invoke_llm(system_prompt, f"Query: {query}")
        
        try:
            clean_text = response_text.replace("```json", "").replace("```", "").strip()
            return json.loads(clean_text)
        except Exception:
            # Fallback logic
            return {
                "intent": "search" if selected_slug and selected_slug != "none" else "chat",
                "strategy": "hybrid" if selected_slug and selected_slug != "none" else "chat",
                "slugs": [selected_slug] if selected_slug and selected_slug != "none" else [],
                "entities": []
            }

    def _invoke_llm(self, system: str, user: str) -> str:
        from langchain_openai import ChatOpenAI
        from langchain_ollama import ChatOllama
        
        messages = [SystemMessage(content=system), HumanMessage(content=user)]
        if self.provider == "openai":
            return ChatOpenAI(model=self.model, temperature=0).invoke(messages).content
        else:
            base_url = os.getenv("OLLAMA_HOST_URL", "http://host.docker.internal:11434")
            return ChatOllama(model=self.model, temperature=0, base_url=base_url).invoke(messages).content