"""
Prompt builder - assembles prompts dynamically based on document types.
"""

import yaml
from pathlib import Path
from typing import Set, Dict, Any


class PromptBuilder:
    """Loads and assembles prompts from YAML config."""

    def __init__(self):
        """Load prompt configuration from YAML."""
        config_path = Path(__file__).parent / "config.yaml"
        with open(config_path, "r") as f:
            self.config: Dict[str, Any] = yaml.safe_load(f)

    def build_system_prompt(
        self,
        available_documents: str,
        doc_types: Set[str],
        use_simple: bool = False
    ) -> str:
        """
        Build system prompt dynamically based on document types.

        Args:
            available_documents: Formatted list of available documents
            doc_types: Set of document types present (e.g., {'book', 'script'})
            use_simple: Use simplified prompt for smaller models

        Returns:
            Assembled system prompt
        """
        if use_simple:
            return self.config["simple"]["template"].format(
                available_documents=available_documents
            )

        sections = []

        # Core sections (always included)
        core = self.config["core"]
        sections.append(core["base"])
        sections.append(core["tool_selection"])
        sections.append(core["slug_finding"])

        # Conditional sections based on doc_types
        doc_type_configs = self.config["doc_types"]
        added_sections = set()  # Avoid duplicates

        for doc_type in doc_types:
            if doc_type in doc_type_configs:
                for section_key, section_text in doc_type_configs[doc_type].items():
                    # Use section_text as unique identifier to prevent duplicates
                    if section_text not in added_sections:
                        sections.append(section_text)
                        added_sections.add(section_text)

        # Citations (always last before documents)
        sections.append(core["citations"])

        # Available documents list
        sections.append(f"\n{available_documents}")

        return "\n\n".join(sections)

    def get_citation_reminder(self) -> str:
        """Get citation reminder for single-document searches."""
        return self.config["reminders"]["citation"]

    def get_comparative_citation_reminder(self) -> str:
        """Get citation reminder for multi-document searches."""
        return self.config["reminders"]["comparative_citation"]

    def get_rephrase_prompt(self, original_query: str, book_title: str = None) -> str:
        """Get query rephrasing prompt."""
        context = f" in the document '{book_title}'" if book_title else ""
        return self.config["rephrase"]["template"].format(
            context=context,
            original_query=original_query
        )
