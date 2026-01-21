"""
MCP Client using LLM providers (OpenAI, Local/Ollama) for function calling.
"""

import asyncio
import ast
import json
import sys
from typing import Optional
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from src.llm.providers import ModelRouter
from src.llm.config import LLMConfig
from src.monitoring.metrics import QueryTimer, NoOpQueryTimer, LLMRelevanceScore
from src.monitoring.judge import ResponseJudge
from src.monitoring.tracer import init_phoenix_tracing, disable_tracing
from src.mcp_client.prompts import (
    get_system_prompt,
    get_citation_reminder,
    get_comparative_citation_reminder,
)


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

        Args:
            provider: LLM provider to use (openai, local, or None for default from config)
            openai_api_key: (DEPRECATED) OpenAI API key for backward compatibility
            model: (DEPRECATED) Model name for backward compatibility
            ephemeral: If True, disable metrics/tracing (conversation only in memory, can still use OpenAI)
            internal_mode: If True, force local LLM only (no external API calls)
        """
        # Set ephemeral mode environment variables BEFORE any metrics/tracing imports
        import os

        if ephemeral:
            os.environ["EPHEMERAL_MODE"] = "true"
            os.environ["DISABLE_TRACING"] = "true"
            disable_tracing()
        else:
            if "EPHEMERAL_MODE" in os.environ:
                del os.environ["EPHEMERAL_MODE"]
            if "DISABLE_TRACING" in os.environ:
                del os.environ["DISABLE_TRACING"]

        if internal_mode:
            provider = "local"
            print("[INFO] Internal mode enabled - forcing local LLM (no external API calls)")

        self.router = ModelRouter()
        self.llm_provider = self.router.get_provider(provider)

        if model:
            self.llm_provider.model = model

        self.config = LLMConfig.from_env()
        self.ephemeral = ephemeral
        self.internal_mode = internal_mode

        if not self.ephemeral:
            init_phoenix_tracing()

        self.session: ClientSession | None = None
        self.stdio_context = None
        self.tools_cache = []
        self.read_stream = None
        self.write_stream = None

        if self.config.enable_judge and not self.ephemeral:
            self.judge = ResponseJudge(llm_provider=self.llm_provider)
            print(
                f"[INFO] Response quality judge enabled using {self.llm_provider.provider_name} provider"
            )
        else:
            self.judge = None

    def _get_max_tokens_for_provider(self) -> int:
        if self.llm_provider.provider_name == "local":
            return 1536
        return 4096

    def _get_temperature_for_provider(self) -> float:
        if self.llm_provider.provider_name == "local":
            return 0.0
        return 0.1

    async def connect_to_mcp_server(self):
        """Connect to the MCP server."""
        import os

        env = dict(os.environ)
        # Use sys.executable to ensure we use the same python interpreter (venv)
        server_params = StdioServerParameters(
            command=sys.executable, args=["-m", "src.mcp_server"], env=env
        )
        self.stdio_context = stdio_client(server_params)
        self.read_stream, self.write_stream = await self.stdio_context.__aenter__()
        self.session = ClientSession(self.read_stream, self.write_stream)
        await self.session.__aenter__()
        await self.session.initialize()
        response = await self.session.list_tools()
        self.tools_cache = self._convert_mcp_tools_to_openai(response.tools)
        print(
            f"Connected to MCP server. Available tools: {[t['function']['name'] for t in self.tools_cache]}"
        )

    def _convert_mcp_tools_to_openai(self, mcp_tools) -> list[dict]:
        openai_tools = []
        for tool in mcp_tools:
            openai_tools.append(
                {
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": tool.inputSchema,
                    },
                }
            )
        return openai_tools

    async def call_mcp_tool(self, tool_name: str, arguments: dict) -> str:
        """Execute a tool call via MCP server with error handling."""
        try:
            if not self.session:
                raise RuntimeError("MCP session not initialized.")

            result = await self.session.call_tool(tool_name, arguments)
            text_content = "\n".join(
                [item.text for item in result.content if hasattr(item, "text")]
            )

            if not text_content:
                return f"Tool {tool_name} returned no content."

            text_lower = text_content.lower()

            # Auto-retry: If search_multiple_documents called with only 1 document, use search_document instead
            if (
                tool_name == "search_multiple_documents"
                and "doc_identifiers" in arguments
                and isinstance(arguments["doc_identifiers"], list)
                and len(arguments["doc_identifiers"]) == 1
                and "validation" in text_lower
                and "too short" in text_lower
            ):
                print(
                    f"[TOOL] Validation error: search_multiple_documents requires 2+ docs, got {len(arguments['doc_identifiers'])}"
                )
                print("[TOOL] Auto-retrying with search_document instead...")

                retry_arguments = {
                    "query": arguments["query"],
                    "doc_identifier": arguments["doc_identifiers"][0],
                    "limit": arguments.get("limit_per_doc", 5),
                }
                return await self.call_mcp_tool("search_document", retry_arguments)

            return text_content
        except Exception as e:
            print(f"[ERROR] Error calling tool '{tool_name}': {str(e)}")
            return str(e)

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

    async def _handle_tool_calls(
        self,
        assistant_message,
        conversation_history: list,
        timer: QueryTimer,
    ) -> list:
        conversation_history.append(
            {
                "role": "assistant",
                "content": assistant_message.content,
                "tool_calls": [
                    {
                        "id": tc["id"],
                        "type": "function",
                        "function": {
                            "name": tc["function"]["name"],
                            "arguments": tc["function"]["arguments"],
                        },
                    }
                    for tc in assistant_message.tool_calls
                ],
            }
        )

        for tool_call in assistant_message.tool_calls:
            function_name = tool_call["function"]["name"]
            timer.add_tool_call(function_name)

            try:
                function_args = json.loads(tool_call["function"]["arguments"])
            except json.JSONDecodeError:
                function_args = {}

            # Normalize parameters
            if "limit_per_doc" in function_args and isinstance(
                function_args["limit_per_doc"], str
            ):
                try:
                    function_args["limit_per_doc"] = int(function_args["limit_per_doc"])
                except ValueError:
                    pass
            if "topk" in function_args and isinstance(function_args["topk"], str):
                try:
                    function_args["topk"] = int(function_args["topk"])
                except ValueError:
                    pass

            # Translate title to slug
            function_args, _ = self._translate_doc_identifier(function_args)
            print(f"[TOOL] Calling: {function_name}({function_args})")

            tool_result = await self.call_mcp_tool(function_name, function_args)

            if function_name == "search_document":
                results_count = self._extract_search_results_count(tool_result)
                timer.set_num_results(results_count)
                if results_count == 0:
                    original_query = function_args.get("query", "")
                    rephrased_query = self._rephrase_query(original_query)
                    if rephrased_query and rephrased_query != original_query:
                        print(f"[RETRY] Rephrased: '{rephrased_query}'")
                        retry_args = function_args.copy()
                        retry_args["query"] = rephrased_query
                        retry_result = await self.call_mcp_tool(
                            function_name, retry_args
                        )
                        retry_count = self._extract_search_results_count(retry_result)
                        timer.set_retry_info(
                            original_query, rephrased_query, retry_count
                        )
                        if retry_count > 0:
                            tool_result = retry_result
                            results_count = retry_count
                            timer.set_num_results(results_count)
                        else:
                            timer.set_fallback_to_context()
                    else:
                        timer.set_fallback_to_context()
            else:
                results_count = -1

            tool_content = tool_result
            if function_name == "search_document" and results_count > 0:
                tool_content += get_citation_reminder()
            elif function_name == "search_multiple_documents" and results_count > 0:
                tool_content += get_comparative_citation_reminder()

            conversation_history.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call["id"],
                    "content": tool_content,
                }
            )
        return conversation_history

    def _translate_doc_identifier(self, function_args: dict) -> tuple[dict, str]:
        title_for_retry = None
        # Support both old and new param names during transition
        for key in ["doc_identifier", "book_identifier"]:
            if key in function_args:
                val = function_args[key]
                if isinstance(val, list) and len(val) > 0:
                    val = val[0]
                if isinstance(val, str):
                    val = val.strip("[]")
                function_args[key] = val
                if (
                    hasattr(self, "title_to_slug")
                    and val.lower() in self.title_to_slug
                ):
                    title_for_retry = val
                    function_args[key] = self.title_to_slug[val.lower()]

        for key in ["doc_identifiers", "book_identifiers"]:
            if key in function_args:
                ids = function_args[key]
                if isinstance(ids, str):
                    try:
                        ids = json.loads(ids)
                    except (json.JSONDecodeError, ValueError):
                        try:
                            ids = ast.literal_eval(ids)
                        except (ValueError, SyntaxError):
                            if "," in ids:
                                ids = [s.strip() for s in ids.split(",")]

                if isinstance(ids, list):
                    translated = []
                    for i in ids:
                        if isinstance(i, str):
                            i = i.strip("[]")
                        if (
                            hasattr(self, "title_to_slug")
                            and i.lower() in self.title_to_slug
                        ):
                            translated.append(self.title_to_slug[i.lower()])
                        else:
                            translated.append(i)
                    function_args[key] = translated
        return function_args, title_for_retry

    def _validate_response_uses_tool_results(
        self,
        response_text: str,
        conversation_history: list,
    ) -> None:
        has_tool_results = any(
            msg.get("role") == "tool" and msg.get("content")
            for msg in conversation_history
        )
        if not has_tool_results:
            return
        hallucination_markers = [
            "not well-known",
            "i couldn't find",
            "unfortunately, i",
            "no results were found",
        ]
        response_lower = response_text.lower()
        found = [m for m in hallucination_markers if m in response_lower]
        if found:
            print(f"[WARN] Hallucination markers detected: {found}")

    def _finalize_response(
        self,
        response_text: str,
        user_message: str,
        conversation_history: list,
        timer: QueryTimer,
    ) -> tuple[str, list, str]:
        conversation_history.append({"role": "assistant", "content": response_text})
        timer.set_response(response_text)
        if self.judge:
            score, reasoning = self.judge.assess_response(user_message, response_text)
            timer.set_llm_assessment(score, reasoning)
        else:
            timer.set_llm_assessment(LLMRelevanceScore.NOT_JUDGED, "Judge disabled")
        return response_text, conversation_history, timer.query_id

    def _prepare_conversation(
        self,
        user_message: str,
        conversation_history: list = None,
    ) -> list:
        available_docs, self.title_to_slug, doc_types = self._get_available_documents()
        use_simple = self.llm_provider.provider_name == "local"
        system_prompt = {
            "role": "system",
            "content": get_system_prompt(
                available_docs, doc_types=doc_types, use_simple=use_simple
            ),
        }
        if not conversation_history:
            conversation_history = [system_prompt]
        elif conversation_history[0].get("role") != "system":
            conversation_history = [system_prompt] + conversation_history
        conversation_history.append({"role": "user", "content": user_message})
        return conversation_history

    def _rephrase_query(self, original_query: str) -> str:
        try:
            response = self.llm_provider.chat_completion(
                messages=[
                    {
                        "role": "system",
                        "content": "You are a search query optimizer. Return ONLY the rephrased query.",
                    },
                    {"role": "user", "content": f"Rephrase: {original_query}"},
                ],
                temperature=0.3,
                max_tokens=50,
            )
            return response.content.strip().strip("'\"")
        except Exception:
            return ""

    def _extract_search_results_count(self, tool_result: str) -> int:
        import re

        match = re.search(r"Found (\d+) results?", tool_result)
        if match:
            return int(match.group(1))
        if "No results found" in tool_result or "0 results" in tool_result:
            return 0
        return -1

    async def chat(
        self,
        user_message: str,
        conversation_history: list = None,
    ) -> tuple[str, list, str]:
        TimerClass = NoOpQueryTimer if self.ephemeral else QueryTimer
        with TimerClass(user_message, None) as timer:
            try:
                if not self.session:
                    await self.connect_to_mcp_server()
                conversation_history = self._prepare_conversation(
                    user_message, conversation_history
                )
                llm_response = await self.llm_provider.chat_completion_async(
                    messages=conversation_history,
                    tools=self.tools_cache,
                    tool_choice="auto",
                    temperature=self._get_temperature_for_provider(),
                    max_tokens=self._get_max_tokens_for_provider(),
                )
                if llm_response.tool_calls:

                    class AM:
                        def __init__(self, c, tc):
                            self.content, self.tool_calls = c, tc

                    conversation_history = await self._handle_tool_calls(
                        AM(llm_response.content, llm_response.tool_calls),
                        conversation_history,
                        timer,
                    )
                    final_response = await self.llm_provider.chat_completion_async(
                        messages=conversation_history,
                        tools=None,
                        temperature=0.5,
                        max_tokens=self._get_max_tokens_for_provider(),
                    )
                    return self._finalize_response(
                        final_response.content,
                        user_message,
                        conversation_history,
                        timer,
                    )
                return self._finalize_response(
                    llm_response.content, user_message, conversation_history, timer
                )
            except Exception as e:
                return str(e), conversation_history or [], timer.query_id

    async def close(self):
        try:
            if self.session:
                await self.session.__aexit__(None, None, None)
            if self.stdio_context:
                await self.stdio_context.__aexit__(None, None, None)
        except Exception:
            pass


# Compatibility alias
BookMateAgent = DocMateAgent


async def main():
    """Test the agent."""
    import os

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY environment variable not set")
        return

    agent = DocMateAgent(openai_api_key=api_key)

    try:
        await agent.connect_to_mcp_server()

        # Test conversation
        print("\n=== Doc Mate Agent ===\n")

        response, history = await agent.chat(
            "What is the book 'Meditations' about? Use the document identifier 'mma'."
        )
        print(f"Agent: {response}\n")

        response, history = await agent.chat(
            "Search for passages about 'death' in the same document.",
            conversation_history=history,
        )
        print(f"Agent: {response}\n")

    finally:
        await agent.close()


if __name__ == "__main__":
    asyncio.run(main())
