"""
Tool handlers for book-related MCP operations.

Separates business logic from MCP protocol definitions.
"""

import logging
from mcp.types import TextContent

from src.flows.book_query import (
    search_book_content,
    get_book_summary,
    get_chapter_summaries,
)
from src.content.store import PgresStore

logger = logging.getLogger(__name__)


class BookToolHandlers:
    """Handles all book-related tool requests."""

    @staticmethod
    def _clean_book_identifier(identifier: str) -> str:
        """Strip brackets from book identifier (e.g., '[dth]' -> 'dth')."""
        return identifier.strip('[]')

    def handle_search_book(self, arguments: dict) -> list[TextContent]:
        """
        Handle single-book search requests.

        Args:
            arguments: Dict with 'query', 'book_identifier', 'limit' (optional)

        Returns:
            List of TextContent with formatted search results
        """
        book_identifier = self._clean_book_identifier(arguments["book_identifier"])

        # Get document type to adjust search depth
        # Conversations need more results for diversity and temporal spread
        store = PgresStore()
        book_id = store._resolve_book_id(book_identifier)
        doc_type = None
        if book_id:
            with store.conn.cursor() as cur:
                cur.execute("SELECT doc_type FROM books WHERE book_id = %s", (book_id,))
                result_row = cur.fetchone()
                if result_row:
                    doc_type = result_row[0]

        # Adjust limit based on document type
        default_limit = 15 if doc_type == "conversation" else 5
        limit = arguments.get("limit", default_limit)

        logger.debug(f"Document type: {doc_type}, using limit: {limit}")

        result = search_book_content(
            query=arguments["query"],
            book_identifier=book_identifier,
            limit=limit,
        )

        # Debug logging
        logger.debug(
            f"Search result for '{arguments['query']}' in '{arguments['book_identifier']}':"
        )
        logger.debug(f"  - num_results: {result['num_results']}")
        logger.debug(f"  - chunk_ids: {result.get('chunk_ids', [])}")
        logger.debug(f"  - error: {result.get('error', 'None')}")

        # Format results as readable text
        if result.get("error"):
            return [TextContent(type="text", text=f"Error: {result['error']}")]

        if result["num_results"] == 0:
            output = (
                f"No results found for '{result['query']}' in book '{result['book']}'."
            )
        else:
            output = (
                f"Found {result['num_results']} results for '{result['query']}' in {result['book']}:\n\n"
            )
            for i, chunk in enumerate(result["chunks"], 1):
                chunk_id = chunk.get("id", "unknown")
                metadata = chunk.get("metadata", {})

                # Extract section number from chunk_id (format: slug_section_chunk_hash)
                section_num = "?"
                if "_" in chunk_id:
                    parts = chunk_id.split("_")
                    if len(parts) >= 2:
                        section_num = (
                            parts[1].lstrip("0") or "0"
                        )  # Remove leading zeros

                # Format citation based on metadata
                # Extract timestamp if available (for conversations)
                # Check multiple possible timestamp fields
                timestamp = (
                    metadata.get("timestamp")
                    or metadata.get("created_at")
                    or metadata.get("timestamp_start")
                )
                timestamp_str = f", Time: {timestamp}" if timestamp else ""

                # For conversations, include speaker information
                speakers = metadata.get("speakers", [])
                speaker = metadata.get("speaker") or metadata.get("author")

                if speakers:
                    speakers_str = ", ".join(speakers)
                    citation = f"[Speakers: {speakers_str}{timestamp_str}, Source: {chunk_id}]"
                elif speaker:
                    citation = f"[Speaker: {speaker}{timestamp_str}, Source: {chunk_id}]"
                else:
                    # Try to use section heading from metadata if available
                    heading = metadata.get("heading", "")
                    if heading and len(heading) < 50:  # Use heading if reasonable length
                        citation = f"[{heading}{timestamp_str}, Source: {chunk_id}]"
                    else:
                        # Fall back to section number
                        # Use generic "Section" instead of assuming "Chapter"
                        citation = f"[Section {section_num}{timestamp_str}, Source: {chunk_id}]"

                output += f"Passage {i} {citation}:\n{chunk['text']}\n\n---\n\n"

        return [TextContent(type="text", text=output)]

    def handle_get_book_summary(self, arguments: dict) -> list[TextContent]:
        """
        Handle book summary requests.

        Args:
            arguments: Dict with 'book_identifier'

        Returns:
            List of TextContent with book summary
        """
        result = get_book_summary(self._clean_book_identifier(arguments["book_identifier"]))
        return [
            TextContent(type="text", text=result["summary"] or "No summary available")
        ]

    def handle_get_chapter_summaries(self, arguments: dict) -> list[TextContent]:
        """
        Handle chapter summaries requests.

        Args:
            arguments: Dict with 'book_identifier'

        Returns:
            List of TextContent with all chapter summaries
        """
        result = get_chapter_summaries(self._clean_book_identifier(arguments["book_identifier"]))

        output = f"Found {result['num_chapters']} chapters:\n\n"
        for ch in result["chapters"]:
            output += f"Chapter {ch['chapter_id']}:\n{ch['summary']}\n\n"

        return [TextContent(type="text", text=output)]

    def handle_search_multiple_books(self, arguments: dict) -> list[TextContent]:
        """
        Handle multi-document comparative search requests.

        Args:
            arguments: Dict with:
                - 'query': Search query to use across all documents
                - 'book_identifiers': List of document slugs to search
                - 'limit_per_book': Optional, number of results per document (default: 3)

        Returns:
            List of TextContent with formatted comparative results
        """
        query = arguments["query"]
        book_identifiers = arguments["book_identifiers"]
        user_limit = arguments.get("limit_per_book")  # User-specified limit (if any)

        logger.info(
            f"Multi-document search: '{query}' across {len(book_identifiers)} documents"
        )

        # Collect results from each book sequentially (thread-safe)
        all_results = []
        total_found = 0
        store = PgresStore()

        for book_id in book_identifiers:
            # Adjust limit based on document type (if user didn't specify)
            if user_limit is None:
                book_id_resolved = store._resolve_book_id(book_id)
                doc_type = None
                if book_id_resolved:
                    with store.conn.cursor() as cur:
                        cur.execute("SELECT doc_type FROM books WHERE book_id = %s", (book_id_resolved,))
                        result_row = cur.fetchone()
                        if result_row:
                            doc_type = result_row[0]

                # Conversations need more results for diversity
                limit_for_this_book = 8 if doc_type == "conversation" else 3
            else:
                limit_for_this_book = user_limit

            logger.debug(f"Searching {book_id} with limit={limit_for_this_book}")

            # Reuse existing search function
            result = search_book_content(
                query=query, book_identifier=book_id, limit=limit_for_this_book
            )

            # Track how many results we found
            num_results = result.get("num_results", 0)
            total_found += num_results

            logger.debug(f"  - {book_id}: {num_results} results")

            # Store result with book info for formatting
            all_results.append({"book": book_id, "result": result})

        # Format combined output with clear document separation
        if total_found == 0:
            output = f"Found 0 results for '{query}' in any of the {len(book_identifiers)} documents searched."
        else:
            output = f"Found {total_found} results - Comparative search for '{query}' across {len(book_identifiers)} documents:\n\n"
            output += "=" * 80 + "\n\n"

            for book_data in all_results:
                book_id = book_data["book"]
                result = book_data["result"]

                # Document header
                output += f"### {book_id.upper()} ###\n\n"

                if result.get("error"):
                    output += f"Error: {result['error']}\n\n"
                elif result["num_results"] == 0:
                    output += "No results found in this document.\n\n"
                else:
                    # Format passages for this book
                    for i, chunk in enumerate(result["chunks"], 1):
                        chunk_id = chunk.get("id", "unknown")
                        metadata = chunk.get("metadata", {})

                        # Extract section number
                        section_num = "?"
                        if "_" in chunk_id:
                            parts = chunk_id.split("_")
                            if len(parts) >= 2:
                                section_num = parts[1].lstrip("0") or "0"

                        # Format citation based on metadata
                        # Extract timestamp if available (for conversations)
                        # Check multiple possible timestamp fields
                        timestamp = (
                            metadata.get("timestamp")
                            or metadata.get("created_at")
                            or metadata.get("timestamp_start")
                        )
                        timestamp_str = f", Time: {timestamp}" if timestamp else ""

                        # For conversations, include speaker information
                        speakers = metadata.get("speakers", [])
                        speaker = metadata.get("speaker") or metadata.get("author")

                        if speakers:
                            speakers_str = ", ".join(speakers)
                            citation = f"[Speakers: {speakers_str}{timestamp_str}, Source: {chunk_id}]"
                        elif speaker:
                            citation = f"[Speaker: {speaker}{timestamp_str}, Source: {chunk_id}]"
                        else:
                            heading = metadata.get("heading", "")
                            if heading and len(heading) < 50:
                                citation = f"[{heading}{timestamp_str}, Source: {chunk_id}]"
                            else:
                                citation = f"[Section {section_num}{timestamp_str}, Source: {chunk_id}]"

                        output += f"Passage {i} {citation}:\n{chunk['text']}\n\n"

                output += "-" * 80 + "\n\n"

        return [TextContent(type="text", text=output)]

    def dispatch(self, tool_name: str, arguments: dict) -> list[TextContent]:
        """
        Dispatch tool call to the appropriate handler method.

        Uses reflection to find handler method by name convention:
        Tool 'search_book' -> method 'handle_search_book'

        Args:
            tool_name: Name of the tool to invoke
            arguments: Arguments passed to the tool

        Returns:
            List of TextContent with tool results or error message
        """
        handler_method_name = f"handle_{tool_name}"
        handler = getattr(self, handler_method_name, None)

        if handler and callable(handler):
            logger.debug(f"Dispatching to {handler_method_name}")
            return handler(arguments)
        else:
            logger.error(f"Unknown tool: {tool_name}")
            return [TextContent(type="text", text=f"Unknown tool: {tool_name}")]
