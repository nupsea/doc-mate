"""
Prompts module - public interface for prompt generation.

This module provides a clean API for generating prompts dynamically
based on document types present in the library.
"""

from .builder import PromptBuilder

# Singleton instance
_builder = PromptBuilder()

# Public API - maintains backward compatibility


def get_system_prompt(
    available_books: str,
    doc_types: set = None,
    use_simple: bool = False
) -> str:
    """
    Get the system prompt with available documents list.

    Args:
        available_books: Formatted list of available documents
        doc_types: Set of document types present (e.g., {'book', 'script'})
                   Defaults to {'book'} if not provided
        use_simple: Use simplified prompt for smaller models

    Returns:
        System prompt string
    """
    if doc_types is None:
        doc_types = {"book"}

    return _builder.build_system_prompt(available_books, doc_types, use_simple)


def get_citation_reminder() -> str:
    """Get the citation reminder for single-document searches."""
    return _builder.get_citation_reminder()


def get_comparative_citation_reminder() -> str:
    """Get the citation reminder for multi-document searches."""
    return _builder.get_comparative_citation_reminder()


def get_rephrase_prompt(original_query: str, book_title: str = None) -> str:
    """Get the query rephrasing prompt."""
    return _builder.get_rephrase_prompt(original_query, book_title)


# Export all public functions
__all__ = [
    "get_system_prompt",
    "get_citation_reminder",
    "get_comparative_citation_reminder",
    "get_rephrase_prompt",
]
