"""
Confidence assessment for retrieval results.

Based on best practices from:
- CRAG (Corrective RAG): Three-tier confidence with corrective actions
- Self-RAG: Multi-stage grading (relevance, support, usefulness)
- Production RAG systems: Query-type aware assessment

Key principle: Confidence reflects whether we have ENOUGH context to answer,
not whether we found specific entities. Broad queries should succeed if we
have relevant results, even without named entity matches.

Safe default: BROAD query type (most lenient, prioritizes having content)
"""

from dataclasses import dataclass
from enum import Enum
from typing import List, Dict, Any, Optional
import re


class ConfidenceLevel(Enum):
    """
    Three-tier confidence system (inspired by CRAG).

    Unlike CRAG, LOW doesn't mean "discard and refuse" - it means
    "caveat heavily and provide what's available."
    """
    HIGH = "high"      # Strong evidence - answer directly, allow synthesis
    MEDIUM = "medium"  # Partial evidence - answer with hedging, note gaps
    LOW = "low"        # Weak evidence - caveat heavily, but still provide what's available


class QueryType(Enum):
    """
    Query classification for confidence adjustment.

    Different query types have different confidence requirements:
    - BROAD: Just needs relevant content (summaries, overviews) - SAFE DEFAULT
    - ENTITY: Needs specific entity present in results
    - INFERENCE: Needs sufficient context for reasoning
    - FACTUAL: Needs precise term matches
    """
    BROAD = "broad"          # Default - most lenient
    ENTITY = "entity"
    INFERENCE = "inference"
    FACTUAL = "factual"


@dataclass
class ConfidenceAssessment:
    """Result of confidence assessment."""
    level: ConfidenceLevel
    score: float  # 0.0 to 1.0
    query_type: QueryType
    signals: Dict[str, float]
    evidence_gaps: List[str]  # Entities/terms not found (mainly for ENTITY queries)
    allow_inference: bool  # Whether query permits grounded reasoning
    coverage_summary: str  # For debugging/logging


# =============================================================================
# Query Type Detection Patterns
# =============================================================================

BROAD_PATTERNS = [
    r'\b(summarize|summarise|summary|overview)\b',
    r'\b(what (happens|happened|is happening|is it about))\b',
    r'\b(tell me about|describe|explain)\b',
    r'\b(what is this|what\'s this|what are these)\b',
    r'\b(main (points|themes|ideas|topics|takeaways))\b',
    r'\b(give me a|provide a|can you give).*(summary|overview)\b',
    r'\b(what do we know about)\b',
    r'\b(brief|briefly)\b',
]

INFERENCE_PATTERNS = [
    r'\b(infer|inference|imply|implies|suggest|suggests|indicates?)\b',
    r'\b(why (do|does|did|would|might|could|is|are|was|were))\b',
    r'\b(what (can|could|might|would) (we|you|one|i) (conclude|infer|deduce|gather))\b',
    r'\b(what does (this|that|it) (mean|suggest|imply|tell us))\b',
    r'\b(based on (this|that|the|these))\b',
    r'\b(reason(s|ing)?|rationale|because)\b',
    r'\b(how (do|does|did|would|could) (you|we) (interpret|understand))\b',
]

ENTITY_PATTERNS = [
    r'\b(who is|who are|who was|who were)\s+[A-Z]',
    r'\b(what is|what are)\s+[A-Z][a-z]+',
    r'\bwhere is\s+[A-Z]',
    r'\b(find|search for|look for)\s+[A-Z]',
    r'\b(tell me about)\s+[A-Z][a-z]+\s+(and|or)\s+[A-Z]',
]

FACTUAL_PATTERNS = [
    r'\b(when did|when does|when was|when were|what time)\b',
    r'\b(how many|how much|how often|how long)\b',
    r'\b(what (date|time|year|day|number|percentage))\b',
    r'\b(exactly|specifically|precisely)\b',
    r'\b(list all|name all|count)\b',
]


def detect_query_type(query: str) -> QueryType:
    """
    Detect the type of query to adjust confidence calculation.

    Priority order: INFERENCE > ENTITY > FACTUAL > BROAD (default)

    Safe default: BROAD - most lenient, prioritizes having content over
    finding specific entities. This prevents the system from refusing
    to answer broad questions like "summarize" or "tell me about."
    """
    query_lower = query.lower()

    # Check inference first - user explicitly asking for reasoning
    for pattern in INFERENCE_PATTERNS:
        if re.search(pattern, query_lower):
            return QueryType.INFERENCE

    # Check entity queries - asking about specific named things
    # Use original case for proper noun detection
    for pattern in ENTITY_PATTERNS:
        if re.search(pattern, query, re.IGNORECASE):
            return QueryType.ENTITY

    # Check factual queries - precise facts needed
    for pattern in FACTUAL_PATTERNS:
        if re.search(pattern, query_lower):
            return QueryType.FACTUAL

    # Check broad patterns explicitly
    for pattern in BROAD_PATTERNS:
        if re.search(pattern, query_lower):
            return QueryType.BROAD

    # SAFE DEFAULT: BROAD
    # Rationale:
    # - Most forgiving weights (result_density matters, not entity_coverage)
    # - Lowest thresholds (easier to reach HIGH/MEDIUM confidence)
    # - Allows synthesis and inference
    # - User can always ask more specific follow-up if answer is too general
    # - Matches production RAG behavior: answer with hedging > refuse
    return QueryType.BROAD


# =============================================================================
# Weight Profiles - Query Type Specific
# =============================================================================

WEIGHT_PROFILES = {
    # BROAD queries (DEFAULT): Having relevant content matters most
    # Used for: summaries, overviews, general questions
    QueryType.BROAD: {
        "result_density": 0.45,      # Do we have content?
        "retrieval_score": 0.35,     # Is it relevant?
        "lexical_overlap": 0.15,     # Does it match query terms?
        "entity_coverage": 0.05,     # Entities barely matter for summaries
    },

    # ENTITY queries: Finding the specific entity is critical
    # Used for: "Who is X?", "What is Y?"
    QueryType.ENTITY: {
        "entity_coverage": 0.45,     # Did we find the entity?
        "retrieval_score": 0.30,     # Is context about that entity?
        "result_density": 0.15,      # Some results needed
        "lexical_overlap": 0.10,     # Less important
    },

    # INFERENCE queries: Need good context to reason over
    # Used for: "Why did X?", "What can we infer?"
    QueryType.INFERENCE: {
        "result_density": 0.40,      # Need enough context to reason
        "retrieval_score": 0.30,     # Context must be relevant
        "entity_coverage": 0.15,     # Entities somewhat matter
        "lexical_overlap": 0.15,     # Topic alignment helps
    },

    # FACTUAL queries: Need precise matches
    # Used for: "When did X?", "How many?"
    QueryType.FACTUAL: {
        "lexical_overlap": 0.35,     # Exact terms matter
        "entity_coverage": 0.25,     # Named entities matter
        "retrieval_score": 0.25,     # Relevance
        "result_density": 0.15,      # At least some results
    },
}

# =============================================================================
# Thresholds - Query Type Specific (Lenient)
# =============================================================================

THRESHOLDS = {
    # BROAD: Very lenient - just need some relevant content
    QueryType.BROAD: {"high": 0.45, "medium": 0.20},

    # ENTITY: Moderate - need the entity but not super strict
    QueryType.ENTITY: {"high": 0.60, "medium": 0.30},

    # INFERENCE: Moderate - need enough context to reason
    QueryType.INFERENCE: {"high": 0.50, "medium": 0.25},

    # FACTUAL: Slightly stricter - need relevant facts
    QueryType.FACTUAL: {"high": 0.55, "medium": 0.30},
}


# =============================================================================
# Main Assessor Class
# =============================================================================

class RetrievalConfidenceAssessor:
    """
    Query-type aware confidence assessment.

    Key design principles:
    1. BROAD is the safe default - lenient thresholds, prioritizes having content
    2. Never refuses entirely - LOW confidence still provides what's available
    3. Allows grounded inference for HIGH/MEDIUM or explicit inference queries
    4. Entity gaps only reported for entity-specific queries
    5. Fast heuristic-based (no extra LLM calls)
    """

    def assess(
        self,
        query: str,
        query_entities: List[str],
        retrieved_chunks: List[Dict[str, Any]],
        retrieval_scores: Optional[List[float]] = None,
        max_expected_results: int = 5,
    ) -> ConfidenceAssessment:
        """
        Assess confidence in retrieval results.

        Args:
            query: Original user query
            query_entities: Entities extracted from query (by router)
            retrieved_chunks: List of chunk dicts with 'text' key
            retrieval_scores: Optional relevance scores (0-1)
            max_expected_results: Expected results for density calculation

        Returns:
            ConfidenceAssessment with level, score, gaps, and metadata
        """
        # Step 1: Detect query type (defaults to BROAD)
        query_type = detect_query_type(query)

        # Step 2: Compute individual signals
        signals = self._compute_signals(
            query, query_entities, retrieved_chunks,
            retrieval_scores, max_expected_results
        )

        # Step 3: Apply query-type specific weights
        weights = WEIGHT_PROFILES[query_type]
        weighted_score = sum(
            signals.get(signal, 0) * weight
            for signal, weight in weights.items()
        )

        # Step 4: Determine confidence level using query-type thresholds
        thresholds = THRESHOLDS[query_type]
        if weighted_score >= thresholds["high"]:
            level = ConfidenceLevel.HIGH
        elif weighted_score >= thresholds["medium"]:
            level = ConfidenceLevel.MEDIUM
        else:
            level = ConfidenceLevel.LOW

        # Step 5: Identify evidence gaps (mainly for entity queries)
        evidence_gaps = self._find_evidence_gaps(
            query_entities, retrieved_chunks, query_type
        )

        # Step 6: Determine if inference is allowed
        # Always allow inference for INFERENCE type
        # Allow for HIGH/MEDIUM confidence on other types
        # Even LOW confidence allows basic synthesis (just with heavy caveats)
        allow_inference = (
            query_type == QueryType.INFERENCE or
            level in [ConfidenceLevel.HIGH, ConfidenceLevel.MEDIUM]
        )

        # Step 7: Generate summary for logging
        coverage_summary = (
            f"Type: {query_type.value} | "
            f"Level: {level.value} ({weighted_score:.2f}) | "
            f"Signals: [{self._format_signals(signals)}] | "
            f"Gaps: {evidence_gaps if evidence_gaps else 'none'}"
        )

        return ConfidenceAssessment(
            level=level,
            score=weighted_score,
            query_type=query_type,
            signals=signals,
            evidence_gaps=evidence_gaps,
            allow_inference=allow_inference,
            coverage_summary=coverage_summary,
        )

    def _compute_signals(
        self,
        query: str,
        query_entities: List[str],
        chunks: List[Dict[str, Any]],
        scores: Optional[List[float]],
        max_results: int,
    ) -> Dict[str, float]:
        """Compute individual confidence signals."""
        signals = {}

        # Combine all text for analysis
        combined_text = " ".join([c.get("text", "") for c in chunks]).lower()
        combined_text = f" {combined_text} "  # Pad for word boundary matching

        # Signal 1: Result Density (do we have content?)
        result_count = len(chunks)
        if result_count == 0:
            signals["result_density"] = 0.0
        else:
            # Lenient: 2+ results = full score, 1 result = 0.7
            if result_count >= 2:
                signals["result_density"] = 1.0
            else:
                signals["result_density"] = 0.7

        # Signal 2: Retrieval Score (are results relevant?)
        if scores and len(scores) > 0:
            # Use top-3 average to reduce noise from lower-ranked results
            top_scores = sorted(scores, reverse=True)[:3]
            signals["retrieval_score"] = sum(top_scores) / len(top_scores)
        else:
            # No scores provided - assume decent relevance if we have results
            signals["retrieval_score"] = 0.65 if result_count > 0 else 0.0

        # Signal 3: Entity Coverage (for entity-specific queries)
        if query_entities:
            found_count = 0
            for entity in query_entities:
                entity_lower = entity.lower()
                # Check exact phrase
                if f" {entity_lower} " in combined_text or entity_lower in combined_text:
                    found_count += 1
                else:
                    # Check if major words are present (partial credit)
                    words = [w for w in re.findall(r'\w+', entity_lower) if len(w) > 3]
                    if words:
                        word_matches = sum(1 for w in words if f" {w} " in combined_text)
                        if word_matches == len(words):
                            found_count += 0.8  # All words found
                        elif word_matches > 0:
                            found_count += 0.4  # Some words found
            signals["entity_coverage"] = min(1.0, found_count / len(query_entities))
        else:
            # No entities to check - high score (don't penalize broad queries)
            signals["entity_coverage"] = 1.0 if result_count > 0 else 0.5

        # Signal 4: Lexical Overlap (does content match query terms?)
        stop_words = {
            "what", "when", "where", "which", "who", "whom", "whose", "how",
            "this", "that", "these", "those", "have", "has", "had",
            "does", "do", "did", "will", "would", "could", "should",
            "the", "a", "an", "is", "are", "was", "were", "been", "be",
            "about", "from", "with", "for", "can", "you", "me", "tell",
            "and", "or", "but", "if", "then", "than", "more", "some",
            "into", "onto", "upon", "also", "just", "only", "very",
        }

        query_words = set(re.findall(r'\w+', query.lower()))
        query_words = {w for w in query_words if len(w) > 2 and w not in stop_words}

        # Remove entity words (already counted separately)
        for entity in query_entities:
            query_words -= set(re.findall(r'\w+', entity.lower()))

        if query_words and combined_text.strip():
            overlap = sum(1 for w in query_words if f" {w}" in combined_text or f"{w} " in combined_text)
            signals["lexical_overlap"] = min(1.0, overlap / len(query_words))
        else:
            # No non-entity words to check - neutral score
            signals["lexical_overlap"] = 0.6

        return signals

    def _find_evidence_gaps(
        self,
        query_entities: List[str],
        chunks: List[Dict[str, Any]],
        query_type: QueryType,
    ) -> List[str]:
        """
        Find entities that were searched but not found.

        Only reports gaps for ENTITY queries. For BROAD/INFERENCE queries,
        missing entities aren't "gaps" - they're just not the focus.
        """
        # Only track gaps for entity-specific queries
        if query_type != QueryType.ENTITY:
            return []

        if not query_entities:
            return []

        combined_text = " ".join([c.get("text", "") for c in chunks]).lower()
        combined_text = f" {combined_text} "

        gaps = []
        for entity in query_entities:
            entity_lower = entity.lower()
            # Check if entity or its significant words are present
            if entity_lower not in combined_text:
                words = [w for w in re.findall(r'\w+', entity_lower) if len(w) > 3]
                if not words or not any(f" {w} " in combined_text for w in words):
                    gaps.append(entity)

        return gaps

    def _format_signals(self, signals: Dict[str, float]) -> str:
        """Format signals for logging."""
        return ", ".join(f"{k[:3]}={v:.2f}" for k, v in sorted(signals.items()))


# =============================================================================
# Convenience Function
# =============================================================================

def assess_retrieval_confidence(
    query: str,
    query_entities: List[str],
    chunks: List[Dict[str, Any]],
    scores: Optional[List[float]] = None,
) -> ConfidenceAssessment:
    """Quick confidence assessment using default assessor."""
    assessor = RetrievalConfidenceAssessor()
    return assessor.assess(query, query_entities, chunks, scores)
