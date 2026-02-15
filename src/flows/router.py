"""
Router logic for determining intent and target documents.
"""

import json
from typing import Dict, Any
from langchain_core.messages import SystemMessage, HumanMessage
from src.content.store import PgresStore
from src.llm.config import LLMConfig
from src.utils.caching import lru_cache_result

class QueryRouter:
    """
    Analyzes user query to determine:
    1. Intent (search, compare, explore, chat)
    2. Target Slugs (list of document identifiers)
    """
    
    def __init__(self, provider: str = "openai", model: str = None):
        self.config = LLMConfig.from_env()
        self.provider = provider
        # Use injected model or configured router model
        self.model = model or self.config.router_model

    @lru_cache_result(maxsize=100)
    def _get_available_docs_context(self) -> str:
        """Fetch available documents for the prompt context."""
        try:
            store = PgresStore()
            docs = store.execute("SELECT slug, title, author, doc_type FROM documents", fetch="all")
            if not docs:
                return "No documents available."
            # Format: - [slug] Title (by Author) [type]
            return "\n".join([f"- [{slug}] {title} (by {author}) [{dtype}]" for slug, title, author, dtype in docs])
        except Exception as e:
            return f"Error fetching docs: {e}"

    def route(self, query: str, selected_slug: str = None, history: list = None) -> Dict[str, Any]:
        """Classifies the query and identifies target documents."""
        doc_list = self._get_available_docs_context()
        
        # Build conversation context from recent history
        conversation_context = ""
        if history and len(history) > 1:
            # Get last 2-3 exchanges for context (skipping system prompt if present)
            # Filter for Human and AI messages only
            relevant_history = [
                msg for msg in history 
                if isinstance(msg, (HumanMessage, SystemMessage)) is False or msg.type in ["human", "ai"]
            ]
            recent = relevant_history[-4:]  # Last 4 messages (2 exchanges)
            context_parts = []
            for msg in recent:
                role = "User" if isinstance(msg, HumanMessage) or msg.type == "human" else "Assistant"
                # Truncate long messages
                content = str(msg.content)
                if len(content) > 200:
                    content = content[:200] + "..."
                context_parts.append(f"{role}: {content}")
            conversation_context = "\n".join(context_parts)

        system_prompt = f"""You are a query router for a document assistant.
        
AVAILABLE DOCUMENTS:
{doc_list}

RECENT CONVERSATION (for context):
{conversation_context if conversation_context else "None - this is the first message"}

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

IMPORTANT:
- If the current query references something from the conversation (e.g., "that", "the same", "more about"), infer the topic from the conversation context.
- For follow-up questions, maintain the same target slugs unless the user explicitly changes topic.
- Extract ALL entities mentioned in both the query AND relevant conversation context.

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
            return ChatOllama(
                model=self.model, 
                temperature=0, 
                base_url=self.config.ollama_base_url
            ).invoke(messages).content