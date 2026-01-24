"""
Doc-Mate Agent using LangGraph for orchestration.
"""

import asyncio
from typing import Optional, List, Dict
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage

from src.llm.providers import ModelRouter
from src.llm.config import LLMConfig
from src.monitoring.metrics import QueryTimer, NoOpQueryTimer, LLMRelevanceScore
from src.monitoring.judge import ResponseJudge
from src.monitoring.tracer import init_phoenix_tracing
from src.mcp_client.prompts import get_system_prompt
from src.flows.agent_graph import agent_graph


class DocMateAgent:
    def __init__(
        self,
        provider: Optional[str] = None,
        openai_api_key: str = None,
        model: str = None,
        ephemeral: bool = False,
        internal_mode: bool = False,
    ):
        """
        Initialize DocMate agent with LLM provider abstraction.
        """
        self.provider_name = provider or "openai"
        if internal_mode:
            self.provider_name = "local"

        self.router = ModelRouter()
        self.llm_provider = self.router.get_provider(self.provider_name)

        if model:
            self.llm_provider.model = model

        self.config = LLMConfig.from_env()
        self.ephemeral = ephemeral
        self.internal_mode = internal_mode

        if not self.ephemeral:
            init_phoenix_tracing()

        if self.config.enable_judge and not self.ephemeral:
            self.judge = ResponseJudge(llm_provider=self.llm_provider)
        else:
            self.judge = None

    def _get_available_documents(self) -> tuple[str, dict, set]:
        """Get available documents from database."""
        try:
            from src.content.store import PgresStore

            store = PgresStore()
            with store.conn.cursor() as cur:
                cur.execute(
                    "SELECT slug, title, author, doc_type FROM documents ORDER BY title"
                )
                docs = cur.fetchall()

            if not docs:
                return "No documents currently available in the library.", {}, {"book"}

            doc_list_str = "\n".join(
                [
                    f"- [{slug}] {title}" + (f" by {author}" if author else "")
                    for slug, title, author, _ in docs
                ]
            )
            title_to_slug = {title.lower(): slug for slug, title, _, _ in docs}
            doc_types = {doc_type for _, _, _, doc_type in docs if doc_type}
            return doc_list_str, title_to_slug, doc_types
        except Exception as e:
            print(f"[WARN] Could not load document list: {e}")
            return "Document list unavailable.", {}, {"book"}

    def _to_langchain_messages(self, history: List[Dict]) -> List[BaseMessage]:
        """Convert list of dicts to LangChain message objects."""
        lc_messages = []
        for msg in history:
            role = msg.get("role")
            content = msg.get("content")
            if role == "system":
                lc_messages.append(SystemMessage(content=content))
            elif role == "user":
                lc_messages.append(HumanMessage(content=content))
            elif role == "assistant":
                lc_messages.append(AIMessage(content=content))
        return lc_messages

    def _finalize_response(
        self,
        response_text: str,
        user_message: str,
        conversation_history: list,
        timer: QueryTimer,
    ) -> tuple[str, list, str]:
        """Add response to history and record metrics."""
        conversation_history.append({"role": "assistant", "content": response_text})
        timer.set_response(response_text)
        if self.judge:
            score, reasoning = self.judge.assess_response(user_message, response_text)
            timer.set_llm_assessment(score, reasoning)
        else:
            timer.set_llm_assessment(LLMRelevanceScore.NOT_JUDGED, "Judge disabled")
        return response_text, conversation_history, timer.query_id

    async def chat(
        self,
        user_message: str,
        conversation_history: list = None,
    ) -> tuple[str, list, str]:
        """
        Send a message and let LangGraph handle the reasoning/tool loop.
        """
        TimerClass = NoOpQueryTimer if self.ephemeral else QueryTimer
        
        with TimerClass(user_message, None) as timer:
            try:
                # 1. Prepare conversation and prompts
                available_docs, _, doc_types = self._get_available_documents()
                use_simple = self.llm_provider.provider_name == "local"
                system_content = get_system_prompt(available_docs, doc_types=doc_types, use_simple=use_simple)
                
                if not conversation_history:
                    conversation_history = [{"role": "system", "content": system_content}]
                elif conversation_history[0].get("role") != "system":
                    conversation_history = [{"role": "system", "content": system_content}] + conversation_history
                
                conversation_history.append({"role": "user", "content": user_message})
                
                # 2. Convert to LangChain format
                messages = self._to_langchain_messages(conversation_history)
                
                # 3. Invoke the Graph
                print(f"[AGENT] Invoking LangGraph with provider={self.llm_provider.provider_name}...")
                result = await agent_graph.ainvoke({
                    "messages": messages,
                    "provider": self.llm_provider.provider_name,
                    "model": self.llm_provider.model,
                    "doc_types": doc_types,
                    "selected_doc_slug": "" # Optional future expansion
                })
                
                # 4. Extract final message
                final_msg = result["messages"][-1].content
                
                # 5. Track tool calls in metrics (optional, for backward compatibility with dashboard)
                for msg in result["messages"]:
                    if isinstance(msg, AIMessage) and msg.tool_calls:
                        for tc in msg.tool_calls:
                            timer.add_tool_call(tc["name"])

                return self._finalize_response(final_msg, user_message, conversation_history, timer)

            except Exception as e:
                import traceback
                traceback.print_exc()
                error_msg = f"Error: {str(e)}"
                return error_msg, conversation_history or [], timer.query_id

    async def connect_to_mcp_server(self):
        """Deprecated compatibility method - no-op in LangGraph."""
        pass

    async def close(self):
        """Deprecated compatibility method - no-op in LangGraph."""
        pass


# Compatibility alias
BookMateAgent = DocMateAgent