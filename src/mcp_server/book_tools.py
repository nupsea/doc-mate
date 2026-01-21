import logging
import sys
import os

# CRITICAL: Configure logging BEFORE any other imports to prevent stdout pollution
# MCP protocol requires stdout to be reserved exclusively for JSON-RPC messages
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    stream=sys.stderr,
    force=True,
)

# Suppress third-party library logging that might output to stdout
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
logging.getLogger("transformers").setLevel(logging.WARNING)
logging.getLogger("torch").setLevel(logging.WARNING)

# Disable transformers progress bars and verbosity
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Now safe to import other modules
from mcp.server import Server
from mcp.types import Tool, TextContent
import mcp.server.stdio

from src.mcp_server.tool_handlers import DocumentToolHandlers

logger = logging.getLogger(__name__)
app = Server("doc-mate-server")

# Initialize tool handlers
tool_handlers = DocumentToolHandlers()


@app.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="search_document",
            description="Search a single document. For comparisons or multiple sources, use 'search_multiple_documents'.",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The search query"},
                    "doc_identifier": {
                        "type": "string",
                        "description": "The document SLUG from the available documents list (e.g., 'abc', 'xyz'). MUST use the slug shown in [square brackets], NOT the full title.",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Number of results to return",
                        "default": 5,
                    },
                },
                "required": ["query", "doc_identifier"],
            },
        ),
        Tool(
            name="get_document_summary",
            description="Get the overall summary of the entire document. Use this when the user asks for a high-level overview of what the whole document is about, its main themes, or general content. Returns a synthesized summary of the complete work.",
            inputSchema={
                "type": "object",
                "properties": {
                    "doc_identifier": {
                        "type": "string",
                        "description": "The document SLUG from the available documents list (e.g., 'abc', 'xyz'). MUST use the slug shown in [square brackets], NOT the full title.",
                    }
                },
                "required": ["doc_identifier"],
            },
        ),
        Tool(
            name="get_document_chapters",
            description="Get summaries of all sections/chapters in a document. Use this when the user wants to see the document structure, understand what each section covers, or get a chapter-by-chapter breakdown. Returns summaries for each section in order.",
            inputSchema={
                "type": "object",
                "properties": {
                    "doc_identifier": {
                        "type": "string",
                        "description": "The document SLUG from the available documents list (e.g., 'abc', 'xyz'). MUST use the slug shown in [square brackets], NOT the full title.",
                    }
                },
                "required": ["doc_identifier"],
            },
        ),
        Tool(
            name="search_multiple_documents",
            description="Search multiple documents (2-5) simultaneously. Required for comparisons ('Compare X and Y') or multi-source queries. More efficient than multiple single searches.",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query to use across all documents. Use specific, concrete terms rather than abstract concepts. For better results with comparative questions, include multiple related terms separated by spaces (e.g., 'concept1 concept2 term1 term2' instead of just 'concept1')"
                    },
                    "doc_identifiers": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of 2-5 document SLUGS to search and compare (e.g., ['abc', 'xyz']). Can mix document types (e.g., compare a book with a conversation). MUST use slugs from [square brackets], NOT titles.",
                        "minItems": 2,
                        "maxItems": 5
                    },
                    "limit_per_doc": {
                        "type": "integer",
                        "description": "Number of results to return from each document",
                        "default": 3,
                        "minimum": 1,
                        "maximum": 5
                    }
                },
                "required": ["query", "doc_identifiers"],
            },
        ),
    ]


@app.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    """
    MCP tool call handler - dispatches to appropriate handler method.

    This is a thin wrapper that delegates all business logic to the
    DocumentToolHandlers class for better modularity and testability.
    """
    return tool_handlers.dispatch(name, arguments)


async def main():
    """Run the MCP server using stdio transport."""
    async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
        await app.run(read_stream, write_stream, app.create_initialization_options())


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
