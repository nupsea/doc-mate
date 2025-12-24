"""Document parsers for different document types."""

from .base import DocumentParser, DocumentType
from .book_parser import BookParser
from .script_parser import ScriptParser
from .conversation_parser import ConversationParser
from .report_parser import ReportParser
from .markdown_parser import MarkdownParser

__all__ = [
    "DocumentParser",
    "DocumentType",
    "BookParser",
    "ScriptParser",
    "ConversationParser",
    "ReportParser",
    "MarkdownParser",
]


def get_parser(file_path: str, doc_type: str, slug: str, split_pattern: str = None):
    """Factory function to get appropriate parser based on document type."""

    parsers = {
        "book": BookParser,
        "tech_doc": MarkdownParser,
        "script": ScriptParser,
        "conversation": ConversationParser,
        "report": ReportParser,
    }

    parser_class = parsers.get(doc_type)
    if not parser_class:
        raise ValueError(f"Unknown document type: {doc_type}")

    return parser_class(file_path, slug, split_pattern=split_pattern)
