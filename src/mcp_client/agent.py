"""
MCP Client using LLM providers (OpenAI, Local/Ollama) for function calling.
"""

import asyncio
import ast
import json
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


class BookMateAgent:
    def __init__(
        self,
        provider: Optional[str] = None,
        openai_api_key: str = None,
        model: str = None,
        ephemeral: bool = False,
        internal_mode: bool = False
    ):
        """
        Initialize BookMate agent with LLM provider abstraction.

        Args:
            provider: LLM provider to use (openai, local, or None for default from config)
            openai_api_key: (DEPRECATED) OpenAI API key for backward compatibility
            model: (DEPRECATED) Model name for backward compatibility
            ephemeral: If True, disable metrics/tracing (conversation only in memory, can still use OpenAI)
            internal_mode: If True, force local LLM only (no external API calls)
        """
        # Set ephemeral mode environment variables BEFORE any metrics/tracing imports
        # This ensures MetricsCollector singleton and Phoenix tracing respect ephemeral mode
        import os
        if ephemeral:
            os.environ["EPHEMERAL_MODE"] = "true"
            os.environ["DISABLE_TRACING"] = "true"  # Disable Phoenix tracing completely
            # Uninstrument any existing Phoenix tracing (if switching from normal/internal mode)
            disable_tracing()
        else:
            # Clear ephemeral flags when switching back to normal mode
            if "EPHEMERAL_MODE" in os.environ:
                del os.environ["EPHEMERAL_MODE"]
            if "DISABLE_TRACING" in os.environ:
                del os.environ["DISABLE_TRACING"]

        # Internal mode forces local provider
        if internal_mode:
            provider = "local"
            print("[INFO] Internal mode enabled - forcing local LLM (no external API calls)")

        # Initialize LLM provider via ModelRouter
        self.router = ModelRouter()
        self.llm_provider = self.router.get_provider(provider)

        # Backward compatibility: set model if provided
        if model:
            self.llm_provider.model = model

        # Load config to check if judge should be enabled
        self.config = LLMConfig.from_env()

        # Privacy mode flags
        self.ephemeral = ephemeral
        self.internal_mode = internal_mode

        # Initialize tracing ONLY if not in ephemeral mode
        if not self.ephemeral:
            init_phoenix_tracing()

        if self.ephemeral and self.internal_mode:
            print("[INFO] Private mode (ephemeral + internal) - no metrics/tracing, local LLM only")
        elif self.ephemeral:
            print("[INFO] Ephemeral mode - no metrics/tracing saved, conversation in memory only")
        elif self.internal_mode:
            print("[INFO] Internal mode - local LLM only, metrics still collected")

        self.session: ClientSession | None = None
        self.stdio_context = None
        self.tools_cache = []
        self.read_stream = None
        self.write_stream = None

        # Initialize judge only if enabled (auto-disabled for local LLM or ephemeral mode)
        if self.config.enable_judge and not self.ephemeral:
            self.judge = ResponseJudge(llm_provider=self.llm_provider)
            print(f"[INFO] Response quality judge enabled using {self.llm_provider.provider_name} provider")
        else:
            self.judge = None
            if self.ephemeral:
                print("[INFO] Response quality judge disabled (ephemeral mode)")
            else:
                print("[INFO] Response quality judge disabled (improves local LLM performance)")

    def _get_max_tokens_for_provider(self) -> int:
        """
        Get appropriate max_tokens based on provider for optimal performance.

        Returns:
            max_tokens value (smaller for local LLM for faster responses)
        """
        if self.llm_provider.provider_name == "local":
            # Balanced token limit for local LLM
            # 1536 is good for llama3.1:8b (better than 3b, can handle more)
            return 1536
        else:
            # OpenAI and other cloud providers can handle more tokens efficiently
            return 4096

    def _get_temperature_for_provider(self) -> float:
        """
        Get optimal temperature for function calling based on provider.

        Returns:
            temperature value (0 for local LLM = more deterministic function calls)
        """
        if self.llm_provider.provider_name == "local":
            # Temperature 0 = more deterministic, better for function calling
            # Small models need maximum predictability
            return 0.0
        else:
            # OpenAI models can handle slight randomness
            return 0.1

    async def connect_to_mcp_server(self):
        """Connect to the MCP server."""
        import os

        # Pass environment variables to MCP server subprocess
        env = dict(os.environ)

        server_params = StdioServerParameters(
            command="python", args=["-m", "src.mcp_server"], env=env
        )

        # Use async context manager correctly
        self.stdio_context = stdio_client(server_params)
        self.read_stream, self.write_stream = await self.stdio_context.__aenter__()

        self.session = ClientSession(self.read_stream, self.write_stream)
        await self.session.__aenter__()
        await self.session.initialize()

        # Fetch available tools from MCP server
        response = await self.session.list_tools()
        self.tools_cache = self._convert_mcp_tools_to_openai(response.tools)

        print(
            f"Connected to MCP server. Available tools: {[t['function']['name'] for t in self.tools_cache]}"
        )

    def _convert_mcp_tools_to_openai(self, mcp_tools) -> list[dict]:
        """Convert MCP tool format to OpenAI function calling format."""
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
                raise RuntimeError(
                    "MCP session not initialized. Call connect_to_mcp_server() first."
                )

            result = await self.session.call_tool(tool_name, arguments)
            # Combine all text content from the response
            text_content = "\n".join(
                [item.text for item in result.content if hasattr(item, "text")]
            )

            if not text_content:
                return f"Tool {tool_name} returned no content."

            # Check if response is a validation error (MCP returns these as text, not exceptions)
            text_lower = text_content.lower()

            # Auto-retry: If search_multiple_books called with only 1 book, use search_book instead
            if (tool_name == "search_multiple_books" and
                "book_identifiers" in arguments and
                isinstance(arguments["book_identifiers"], list) and
                len(arguments["book_identifiers"]) == 1 and
                "validation" in text_lower and "too short" in text_lower):

                print(f"[TOOL] Validation error: search_multiple_books requires 2+ books, got {len(arguments['book_identifiers'])}")
                print("[TOOL] Auto-retrying with search_book instead...")

                # Convert to search_book call
                retry_arguments = {
                    "query": arguments["query"],
                    "book_identifier": arguments["book_identifiers"][0],
                    "limit": arguments.get("limit_per_book", 5)
                }

                # Recursive call with corrected tool
                return await self.call_mcp_tool("search_book", retry_arguments)

            return text_content
        except Exception as e:
            error_msg = f"Error calling tool '{tool_name}': {str(e)}"
            print(f"[ERROR] {error_msg}")
            return error_msg

    def _get_available_books(self) -> tuple[str, dict, set]:
        """
        Get available documents from database.

        Returns:
            (formatted_list, title_to_slug_map, doc_types)
        """
        try:
            from src.content.store import PgresStore

            store = PgresStore()
            with store.conn.cursor() as cur:
                cur.execute("SELECT slug, title, author, doc_type FROM books ORDER BY title")
                books = cur.fetchall()

            if not books:
                return "No documents currently available in the library.", {}, {"book"}

            # Create list with slugs (needed for tool calls)
            book_list = "\n".join(
                [
                    f"- [{slug}] {title}" + (f" by {author}" if author else "")
                    for slug, title, author, _ in books
                ]
            )

            # Create mapping for internal use
            title_to_slug = {title.lower(): slug for slug, title, _, _ in books}

            # Collect unique doc types
            doc_types = {doc_type for _, _, _, doc_type in books if doc_type}
            if not doc_types:
                doc_types = {"book"}  # Default if no doc_type set

            return book_list, title_to_slug, doc_types
        except Exception as e:
            print(f"[WARN] Could not load document list: {e}")
            return "Document list unavailable.", {}, {"book"}

    async def _handle_tool_calls(
        self,
        assistant_message,
        conversation_history: list,
        timer: QueryTimer
    ) -> list:
        """
        Handle all tool call execution.

        Returns:
            Updated conversation history with tool results
        """
        # Add assistant's tool call request to history
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

        # Execute each tool call via MCP
        for tool_call in assistant_message.tool_calls:
            function_name = tool_call["function"]["name"]

            # Track tool call
            timer.add_tool_call(function_name)

            try:
                function_args = json.loads(tool_call["function"]["arguments"])
            except json.JSONDecodeError as e:
                print(f"[ERROR] Invalid JSON in tool arguments: {e}")
                function_args = {}

            # Normalize numeric parameters (LLM sometimes returns strings)
            if "limit_per_book" in function_args and isinstance(function_args["limit_per_book"], str):
                try:
                    function_args["limit_per_book"] = int(function_args["limit_per_book"])
                except ValueError:
                    pass  # Keep as string if can't convert

            if "topk" in function_args and isinstance(function_args["topk"], str):
                try:
                    function_args["topk"] = int(function_args["topk"])
                except ValueError:
                    pass

            # Translate book title to slug if needed
            function_args, _ = self._translate_book_identifier(function_args)

            print(f"[TOOL] Calling: {function_name}({function_args})")

            # Call MCP server (already has error handling)
            tool_result = await self.call_mcp_tool(
                function_name, function_args
            )

            # Track search results for metrics
            if function_name == "search_book":
                results_count = self._extract_search_results_count(tool_result)
                timer.set_num_results(results_count)
                print(f"[SEARCH] Found {results_count} results")

                # Retry logic: if no results, rephrase and try once more
                if results_count == 0:
                    original_query = function_args.get("query", "")
                    rephrased_query = self._rephrase_query(original_query)

                    if rephrased_query and rephrased_query != original_query:
                        print(f"[RETRY] Original: '{original_query}'")
                        print(f"[RETRY] Rephrased: '{rephrased_query}'")

                        # Retry search with rephrased query
                        retry_args = function_args.copy()
                        retry_args["query"] = rephrased_query
                        retry_result = await self.call_mcp_tool(function_name, retry_args)
                        retry_count = self._extract_search_results_count(retry_result)

                        print(f"[RETRY] Found {retry_count} results")

                        # Track retry in metrics
                        timer.set_retry_info(original_query, rephrased_query, retry_count)

                        # Use retry result if it found something
                        if retry_count > 0:
                            tool_result = retry_result
                            results_count = retry_count
                            timer.set_num_results(results_count)
                            print("[RETRY] Success - using retry results")
                        else:
                            print("[RETRY] No improvement - LLM will use context knowledge")
                            timer.set_fallback_to_context()
                    else:
                        # Couldn't rephrase, fall back to context
                        print("[RETRY] Could not rephrase query - LLM will use context knowledge")
                        timer.set_fallback_to_context()
            else:
                results_count = -1
                print(f"[TOOL] Completed: {function_name}")

            print()

            # Add tool result to conversation with citation reminder
            tool_content = tool_result
            if function_name == "search_book" and results_count > 0:
                tool_content += get_citation_reminder()
            elif function_name == "search_multiple_books" and results_count > 0:
                tool_content += get_comparative_citation_reminder()

            conversation_history.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call["id"],
                    "content": tool_content,
                }
            )

        return conversation_history

    def _translate_book_identifier(self, function_args: dict) -> tuple[dict, str]:
        """
        Translate book title(s) to slug(s) if needed.

        Handles both:
        - book_identifier (single book)
        - book_identifiers (multiple books for comparative search)

        Returns:
            (updated_function_args, book_title_for_retry)
        """
        book_title_for_retry = None

        # Handle single book identifier
        if "book_identifier" in function_args:
            book_id = function_args["book_identifier"]

            # Normalize: LLM sometimes returns list instead of string
            if isinstance(book_id, list) and len(book_id) > 0:
                book_id = book_id[0]
                function_args["book_identifier"] = book_id
                print(f"[TOOL] Normalized list to string: {book_id}")

            # Strip brackets if present (LLM may extract [slug] from document list)
            if isinstance(book_id, str):
                book_id = book_id.strip("[]")
                function_args["book_identifier"] = book_id

            print(f"[TOOL] LLM provided book_identifier: '{book_id}'")
            print(
                f"[TOOL] Available mappings: {self.title_to_slug if hasattr(self, 'title_to_slug') else 'NONE'}"
            )

            # Try to match as title first (case-insensitive)
            if (
                hasattr(self, "title_to_slug")
                and book_id.lower() in self.title_to_slug
            ):
                original_id = book_id
                function_args["book_identifier"] = self.title_to_slug[
                    book_id.lower()
                ]
                book_title_for_retry = original_id  # Store title for retry context
                print(
                    f"[TOOL] Translated '{original_id}' -> '{function_args['book_identifier']}'"
                )
            else:
                print(f"[TOOL] NO TRANSLATION - passing '{book_id}' as-is")

        # Handle multiple book identifiers (for comparative search)
        if "book_identifiers" in function_args:
            book_ids = function_args["book_identifiers"]

            # Normalize: LLM sometimes returns string representation instead of list
            if isinstance(book_ids, str):
                try:
                    # Try JSON parse: '["mam", "ili"]'
                    book_ids = json.loads(book_ids)
                    print(f"[TOOL] Normalized JSON string to list: {book_ids}")
                except (json.JSONDecodeError, ValueError):
                    try:
                        # Try Python literal eval: "['mam', 'ili']"
                        book_ids = ast.literal_eval(book_ids)
                        print(f"[TOOL] Normalized Python literal to list: {book_ids}")
                    except (ValueError, SyntaxError):
                        # Last resort: comma-separated string "mam, ili, ody"
                        if ',' in book_ids:
                            book_ids = [s.strip() for s in book_ids.split(',')]
                            print(f"[TOOL] Normalized comma-separated to list: {book_ids}")
                        else:
                            print(f"[ERROR] Could not parse book_identifiers: {book_ids}")
                            return function_args, book_title_for_retry

                function_args["book_identifiers"] = book_ids

            print(f"[TOOL] LLM provided book_identifiers: {book_ids}")

            translated_ids = []
            for book_id in book_ids:
                # Strip brackets if present (LLM may extract [slug] from document list)
                if isinstance(book_id, str):
                    book_id = book_id.strip("[]")

                if (
                    hasattr(self, "title_to_slug")
                    and book_id.lower() in self.title_to_slug
                ):
                    slug = self.title_to_slug[book_id.lower()]
                    translated_ids.append(slug)
                    print(f"[TOOL] Translated '{book_id}' -> '{slug}'")
                else:
                    # Keep as-is if not found (might already be a slug)
                    translated_ids.append(book_id)
                    print(f"[TOOL] NO TRANSLATION for '{book_id}' - passing as-is")

            function_args["book_identifiers"] = translated_ids

        return function_args, book_title_for_retry

    def _validate_response_uses_tool_results(
        self,
        response_text: str,
        conversation_history: list
    ) -> None:
        """
        Validate that the response appears to use tool results rather than general knowledge.
        Logs a warning if the response contains hallucination markers.
        """
        # Check if tools were called (look for tool results in history)
        has_tool_results = any(
            msg.get("role") == "tool" and msg.get("content")
            for msg in conversation_history
        )

        if not has_tool_results:
            return  # No tool results to validate against

        # Hallucination markers that suggest LLM is inferring instead of citing
        hallucination_markers = [
            "not well-known",
            "not widely available",
            "i couldn't find",
            "i made an error in my previous response",
            "unfortunately, i",
            "i don't have information",
            "the search returned no",
            "no results were found",
            "appears to be a",
            "seems to be a",
            "likely",
            "probably",
            "suggests that",
            "indicating that",
        ]

        response_lower = response_text.lower()
        found_markers = [marker for marker in hallucination_markers if marker in response_lower]

        if found_markers:
            print(f"[WARN] Response may be ignoring tool results. Hallucination markers detected: {found_markers}")
            print(f"[WARN] Response preview: {response_text[:200]}...")

    def _finalize_response(
        self,
        response_text: str,
        user_message: str,
        conversation_history: list,
        timer: QueryTimer
    ) -> tuple[str, list, str]:
        """
        Add response to history and assess quality.

        Returns:
            (response_text, updated_conversation_history, query_id)
        """
        conversation_history.append({"role": "assistant", "content": response_text})
        timer.set_response(response_text)

        # Run judge assessment only if enabled
        if self.judge:
            score, reasoning = self.judge.assess_response(user_message, response_text)
            timer.set_llm_assessment(score, reasoning)
        else:
            # Skip judge for performance (local LLM)
            timer.set_llm_assessment(LLMRelevanceScore.NOT_JUDGED, "Judge disabled for performance")

        return response_text, conversation_history, timer.query_id

    def _prepare_conversation(self, user_message: str, conversation_history: list = None) -> list:
        """
        Prepare conversation history with system prompt and user message.
        Automatically truncates history if it exceeds token limit.

        Returns:
            Updated conversation history
        """
        import tiktoken

        MAX_HISTORY_TOKENS = 50000  # Reserve ~50k tokens for conversation history

        # Get available books for system prompt
        available_books, self.title_to_slug, doc_types = self._get_available_books()

        # Create system prompt (use simple version for local LLM)
        use_simple_prompt = self.llm_provider.provider_name == "local"
        system_prompt = {
            "role": "system",
            "content": get_system_prompt(available_books, doc_types=doc_types, use_simple=use_simple_prompt),
        }

        # Check if this is a new conversation or needs system prompt
        if not conversation_history:
            print("[CHAT] Creating NEW conversation")
            conversation_history = [system_prompt]
        else:
            # Ensure system prompt exists in continuing conversations
            if not conversation_history or conversation_history[0].get("role") != "system":
                print("[CHAT] CONTINUING conversation - prepending system prompt")
                conversation_history = [system_prompt] + conversation_history
            else:
                print("[CHAT] CONTINUING conversation with existing system prompt")

            # Truncate old messages if history is too long
            if len(conversation_history) > 2:
                enc = tiktoken.get_encoding("cl100k_base")

                # Count tokens in conversation
                total_tokens = sum(
                    len(enc.encode(str(msg.get("content", ""))))
                    for msg in conversation_history
                )

                if total_tokens > MAX_HISTORY_TOKENS:
                    print(f"[CHAT] History too long ({total_tokens} tokens), truncating...")

                    # Remove old messages (keep system prompt at index 0)
                    while total_tokens > MAX_HISTORY_TOKENS and len(conversation_history) > 2:
                        removed_msg = conversation_history.pop(1)
                        removed_tokens = len(enc.encode(str(removed_msg.get("content", ""))))
                        total_tokens -= removed_tokens

                    print(f"[CHAT] Truncated to {len(conversation_history)} messages ({total_tokens} tokens)")

        print(f"[CHAT] Title-to-slug mapping: {self.title_to_slug}")

        # Add user message
        conversation_history.append({"role": "user", "content": user_message})

        return conversation_history


    def _rephrase_query(self, original_query: str) -> str:
        """
        Use LLM to rephrase a query for better search results.

        Args:
            original_query: The original search query that returned no results

        Returns:
            Rephrased query, or empty string if rephrasing fails
        """
        try:
            response = self.llm_provider.chat_completion(
                messages=[
                    {
                        "role": "system",
                        "content": "You are a search query optimizer. Rephrase the user's query to improve search results. Use simpler terms, remove stop words, and focus on key concepts. Return ONLY the rephrased query, no explanation."
                    },
                    {
                        "role": "user",
                        "content": f"Rephrase this search query: {original_query}"
                    }
                ],
                temperature=0.3,
                max_tokens=50
            )

            rephrased = response.content.strip()
            # Remove quotes if LLM added them
            rephrased = rephrased.strip('"\'')
            return rephrased

        except Exception as e:
            print(f"[RETRY] Error rephrasing query: {e}")
            return ""

    def _extract_search_results_count(self, tool_result: str) -> int:
        """
        Parse the number of results from a search_book tool response.

        Returns:
            Number of results, or -1 if unable to parse
        """
        try:
            # Look for "Found X results" pattern
            import re

            match = re.search(r"Found (\d+) results?", tool_result)
            if match:
                return int(match.group(1))

            # Alternative: Look for "No results found"
            if "No results found" in tool_result or "0 results" in tool_result:
                return 0

            return -1  # Unknown
        except Exception as e:
            print(f"[RETRY] Error parsing search results: {e}")
            return -1

    async def chat(
        self, user_message: str, conversation_history: list = None
    ) -> tuple[str, list, str]:
        """
        Send a message and handle tool calls automatically.

        Returns:
            (assistant_response, updated_conversation_history, query_id)
        """
        print(f"\n{'='*80}")
        print("[CHAT] NEW REQUEST")
        print(f"[CHAT] User message: {user_message}")
        print(
            f"[CHAT] Conversation history length: {len(conversation_history) if conversation_history else 0}"
        )
        print(f"{'='*80}\n")

        # Start monitoring (use no-op timer in ephemeral mode)
        TimerClass = NoOpQueryTimer if self.ephemeral else QueryTimer
        with TimerClass(user_message, None) as timer:
            try:
                if not self.session:
                    raise RuntimeError(
                        "MCP session not initialized. Call connect_to_mcp_server() first."
                    )

                if not user_message or not user_message.strip():
                    raise ValueError("User message cannot be empty.")

                # Prepare conversation with system prompt and user message
                conversation_history = self._prepare_conversation(
                    user_message, conversation_history
                )

                print("[CHAT] Full conversation being sent to LLM:")
                for i, msg in enumerate(conversation_history):
                    role = msg.get("role", "unknown")
                    content = msg.get("content", "")
                    if role == "system":
                        print(f"  [{i}] SYSTEM: {content[:200]}...")
                    elif role == "user":
                        print(f"  [{i}] USER: {content}")
                    elif role == "assistant":
                        print(
                            f"  [{i}] ASSISTANT: {content[:100] if content else '<tool_calls>'}"
                        )
                    elif role == "tool":
                        print(f"  [{i}] TOOL: {content[:100]}...")
                print()

                # Call LLM provider with function calling
                # Use temperature=0 for local LLM to improve function calling reliability
                llm_response = await self.llm_provider.chat_completion_async(
                    messages=conversation_history,
                    tools=self.tools_cache,
                    tool_choice="auto",
                    temperature=self._get_temperature_for_provider(),
                    max_tokens=self._get_max_tokens_for_provider(),
                )

                # Extract tool_calls from unified response
                assistant_message_content = llm_response.content
                assistant_tool_calls = llm_response.tool_calls

                # Check if the model wants to call tools
                if assistant_tool_calls:
                    class AssistantMessage:
                        def __init__(self, content, tool_calls):
                            self.content = content
                            self.tool_calls = tool_calls if tool_calls else []

                    assistant_message = AssistantMessage(
                        content=assistant_message_content,
                        tool_calls=assistant_tool_calls
                    )

                    conversation_history = await self._handle_tool_calls(
                        assistant_message, conversation_history, timer
                    )

                    print("[CHAT] Generating final response from tool results...")
                    print(f"[CHAT] Conversation history for final call: {len(conversation_history)} messages")
                    for i, msg in enumerate(conversation_history[-3:]):  # Show last 3 messages
                        role = msg.get("role", "unknown")
                        content_preview = str(msg.get("content", ""))[:150]
                        print(f"  [{i}] {role}: {content_preview}...")

                    # For final response generation, we need to prevent the model from outputting more tool calls
                    # Small models especially can get confused by seeing tool_calls in history and continue that pattern
                    # Solution:
                    # 1. Explicitly pass tools=None to disable function calling
                    # 2. Use slightly higher temperature to break pattern-copying behavior
                    # 3. Provide clear instruction about expected output format

                    final_temp = 0.3 if self.llm_provider.provider_name == "local" else 0.5

                    final_llm_response = await self.llm_provider.chat_completion_async(
                        messages=conversation_history,
                        tools=None,  # Explicitly disable tool calling for final response
                        temperature=final_temp,  # Higher temp to avoid pattern copying
                        max_tokens=self._get_max_tokens_for_provider(),
                    )

                    final_message = final_llm_response.content
                    print(f"[CHAT] Final response length: {len(final_message)} chars")
                    print(f"[CHAT] Final response: {final_message}")

                    # Validate that response uses tool results
                    self._validate_response_uses_tool_results(final_message, conversation_history)

                    return self._finalize_response(
                        final_message, user_message, conversation_history, timer
                    )

                else:
                    # No tool calls, just return the response
                    response_text = assistant_message_content

                    # Validate response (will log warning if LLM should have called tools)
                    self._validate_response_uses_tool_results(response_text, conversation_history)

                    return self._finalize_response(
                        response_text, user_message, conversation_history, timer
                    )

            except Exception as e:
                error_msg = f"Error during chat: {str(e)}"
                print(f"[ERROR] {error_msg}")
                timer.set_response(error_msg)
                return (
                    error_msg,
                    conversation_history if conversation_history else [],
                    timer.query_id,
                )

    async def close(self):
        """Close the MCP session and cleanup resources."""
        try:
            if self.session:
                await self.session.__aexit__(None, None, None)
                self.session = None
        except Exception as e:
            print(f"Warning: Error closing session: {e}")

        try:
            if self.stdio_context:
                await self.stdio_context.__aexit__(None, None, None)
                self.stdio_context = None
        except Exception as e:
            print(f"Warning: Error closing stdio context: {e}")


async def main():
    """Test the agent."""
    import os

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY environment variable not set")
        return

    agent = BookMateAgent(api_key)

    try:
        await agent.connect_to_mcp_server()

        # Test conversation
        print("\n=== Doc Mate Agent ===\n")

        response, history = await agent.chat(
            "What is the book 'Meditations' about? Use the book identifier 'mma'."
        )
        print(f"Agent: {response}\n")

        response, history = await agent.chat(
            "Search for passages about 'death' in the same book.",
            conversation_history=history,
        )
        print(f"Agent: {response}\n")

    finally:
        await agent.close()


if __name__ == "__main__":
    asyncio.run(main())
