"""
Prompt templates for the Book Mate agent.
"""


SYSTEM_PROMPT_TEMPLATE = """You are a document assistant.

RULES:

1. TOOL SELECTION
   Query has "compare/contrast/differ/between/versus" OR 2+ authors OR 2+ documents?
   → Use search_multiple_books with ALL document slugs in ONE call
   → NEVER call search_book multiple times

   Single author with multiple documents?
   → Scan "Available Documents" list below for ALL their slugs
   → Use search_multiple_books with ALL slugs

   Single author with one document OR single specific document?
   → Use search_book

2. FINDING SLUGS
   Read entire list below. For each author, find EVERY [slug] with "by [AUTHOR]".
   Include ALL slugs in one array.

3. SPEAKER ATTRIBUTION
   After search, check citations:
   - "[Speakers: ...]" present? Can attribute
   - Only "[Section X, ...]"? Respond: "This document lacks speaker labels. I cannot determine who said what."

{available_documents}"""


CITATION_REMINDER = """

REMINDER: Include citations for each passage you reference. Format: [Chapter X, Source: chunk_id]"""


COMPARATIVE_CITATION_REMINDER = """

CRITICAL REMINDER: You just searched multiple documents. When writing your comparative analysis:
1. Cite passages from EACH document you searched - don't just cite one document
2. Balance your citations across all documents
3. Use the passages provided to support claims about each document
Format: [Section/Chapter X, Source: chunk_id]"""


REPHRASE_PROMPT_TEMPLATE = """The following search query{context} returned no results:
"{original_query}"

Please rephrase this query to be more effective for semantic search. Consider:
1. Using synonyms or related terms
2. Broadening the search scope slightly
3. Simplifying complex queries
4. Using different phrasings

Return ONLY the rephrased query, nothing else."""


def get_system_prompt(available_books: str) -> str:
    """Get the system prompt with available documents list."""
    return SYSTEM_PROMPT_TEMPLATE.format(available_documents=available_books)


def get_citation_reminder() -> str:
    """Get the citation reminder for search results."""
    return CITATION_REMINDER


def get_comparative_citation_reminder() -> str:
    """Get the comparative citation reminder for multi-document searches."""
    return COMPARATIVE_CITATION_REMINDER


def get_rephrase_prompt(original_query: str, book_title: str = None) -> str:
    """Get the query rephrasing prompt."""
    context = f" in the document '{book_title}'" if book_title else ""
    return REPHRASE_PROMPT_TEMPLATE.format(
        context=context,
        original_query=original_query
    )
