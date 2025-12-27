"""Simple query classifier - detect if query needs 8B model."""

def needs_complex_model(query: str) -> bool:
    """
    Check if query needs llama3.1:8b (complex) or can use llama3.2:3b (fast).

    Returns True only for: comparisons, multi-document queries
    Returns False (use fast 3b) for everything else
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

    # Default to fast model
    return False
