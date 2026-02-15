"""Simple query classifier - detect if query needs 8B model."""

def needs_complex_model(query: str) -> bool:
    """
    Check if query needs llama3.1:8b (complex) or can use llama3.2:3b (fast).

    Returns True for: comparisons, multi-document queries, relationship/entity queries
    Returns False (use fast 3b) for simple factual lookups
    """
    query_lower = query.lower()

    # Complex patterns requiring 8B model
    complex_keywords = ['compare', 'contrast', 'versus', 'vs.', 'vs ', 'differ']

    # Check for comparison keywords
    for keyword in complex_keywords:
        if keyword in query_lower:
            return True

    # Check for "between X and Y" pattern
    if 'between' in query_lower and ' and ' in query_lower:
        return True

    # Relationship / entity queries need better instruction-following
    relationship_keywords = [
        'connected', 'related', 'relationship', 'interact',
        'who is', 'who are', 'who was', 'who were',
        'tell me about', 'what role',
    ]
    for keyword in relationship_keywords:
        if keyword in query_lower:
            return True

    # "How" + entity-style questions (e.g. "How are X and Y connected?")
    if query_lower.startswith('how') and ' and ' in query_lower:
        return True

    # Default to fast model
    return False
