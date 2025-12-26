"""
LLM-based response quality assessment.
"""

import json
from typing import Optional
from src.llm.providers import LLMProvider, ModelRouter
from src.monitoring.metrics import LLMRelevanceScore


class ResponseJudge:
    """Judges response quality using LLM."""

    ASSESSMENT_PROMPT = """You are evaluating the quality of an AI assistant's response to a user query.

User Query: {query}

Assistant Response: {response}

Evaluate how well the response addresses the user's query. Consider:
1. Did it answer what was asked?
2. Is the information accurate and relevant?
3. Is it complete and helpful?

Provide your assessment in JSON format:
{{
    "score": "EXCELLENT" | "ADEQUATE" | "POOR",
    "reasoning": "Brief explanation of your assessment (1-2 sentences)"
}}

Score definitions:
- EXCELLENT: Fully addresses the query with comprehensive, relevant information
- ADEQUATE: Addresses the query but could be more complete or relevant
- POOR: Does not adequately address the query or contains irrelevant information"""

    def __init__(self, llm_provider: Optional[LLMProvider] = None):
        """
        Initialize response judge with LLM provider.

        Args:
            llm_provider: LLM provider instance (defaults to ModelRouter default)
        """
        if llm_provider is None:
            router = ModelRouter()
            llm_provider = router.get_provider()
        self.llm_provider = llm_provider

    def assess_response(
        self, query: str, response: str
    ) -> tuple[LLMRelevanceScore, Optional[str]]:
        """
        Assess response quality using LLM.

        Returns:
            (score, reasoning)
        """
        try:
            assessment_result = self.llm_provider.chat_completion(
                messages=[
                    {
                        "role": "system",
                        "content": "You are an objective evaluator of AI responses.",
                    },
                    {
                        "role": "user",
                        "content": self.ASSESSMENT_PROMPT.format(
                            query=query, response=response
                        ),
                    },
                ],
                temperature=0.3,
                response_format={"type": "json_object"},
            )

            result_text = assessment_result.content

            # Try to parse JSON - handle cases where Ollama adds extra text
            try:
                result = json.loads(result_text)
            except json.JSONDecodeError:
                # Try to extract JSON object from text
                import re
                json_match = re.search(r'\{[^}]+\}', result_text, re.DOTALL)
                if json_match:
                    result = json.loads(json_match.group(0))
                else:
                    print(f"[WARN] Could not parse JSON from judge response: {result_text[:200]}")
                    return LLMRelevanceScore.NOT_JUDGED, "Could not parse assessment JSON"

            score_str = result.get("score", "NOT_JUDGED").upper()
            reasoning = result.get("reasoning", "")

            # Map string to enum
            score_map = {
                "EXCELLENT": LLMRelevanceScore.EXCELLENT,
                "ADEQUATE": LLMRelevanceScore.ADEQUATE,
                "POOR": LLMRelevanceScore.POOR,
            }

            score = score_map.get(score_str, LLMRelevanceScore.NOT_JUDGED)

            return score, reasoning

        except Exception as e:
            print(f"[ERROR] Failed to assess response: {e}")
            return LLMRelevanceScore.NOT_JUDGED, f"Assessment failed: {str(e)}"
