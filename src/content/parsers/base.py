"""
Base parser interface for all document types.

Philosophy:
- Minimal parsing: Extract only reliable, concrete data
- Let LLM interpret semantics during retrieval
- Type-specific metadata in JSONB, not hardcoded fields
"""

from abc import ABC, abstractmethod
from enum import Enum
from typing import List, Dict, Optional
from pathlib import Path
import hashlib


class DocumentType(Enum):
    """Supported document types."""
    BOOK = "book"
    SCRIPT = "script"
    CONVERSATION = "conversation"
    TECH_DOC = "tech_doc"
    REPORT = "report"


class DocumentParser(ABC):
    """
    Abstract base parser for all document types.

    All parsers must implement:
    - read_content(): Read raw text from file
    - parse(): Extract document structure
    - chunk(): Convert to RAG-ready chunks

    Optional methods:
    - extract_metadata(): Document-level metadata
    - extract_assets(): Images, tables, code blocks
    """

    def __init__(self, file_path: str, slug: str, split_pattern: str = None):
        """
        Initialize parser.

        Args:
            file_path: Path to document file
            slug: Unique identifier for document
            split_pattern: Optional regex pattern for splitting (doc-type specific)
        """
        self.file_path = Path(file_path)
        self.slug = slug
        self.split_pattern = split_pattern
        self.doc_type = self._get_doc_type()

        # Validate file exists
        if not self.file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

    @abstractmethod
    def _get_doc_type(self) -> DocumentType:
        """
        Return the document type this parser handles.

        Returns:
            DocumentType enum value

        Example:
            return DocumentType.SCRIPT
        """
        pass

    @abstractmethod
    def read_content(self) -> str:
        """
        Read raw content from file.

        Returns:
            Raw text content as string

        Implementation notes:
        - For PDFs: Use PyMuPDF or pdfplumber
        - For text files: Use Path.read_text()
        - For special formats: Implement custom reader
        """
        pass

    @abstractmethod
    def parse(self) -> List[Dict]:
        """
        Parse document into semantic units.

        Returns:
            List of dictionaries with parsed sections/scenes/chunks.
            Structure varies by document type.

        Examples:
            Books:         [{"chapter": 1, "title": "...", "text": "..."}, ...]
            Scripts:       [{"scene_number": 1, "heading": "INT. CAFE - DAY", "text": "..."}, ...]
            Conversations: [{"speaker": "Alice", "turn": 1, "text": "..."}, ...]
            Tech Docs:     [{"section": "3.2", "heading": "...", "text": "...", "has_code": True}, ...]
        """
        pass

    @abstractmethod
    def chunk(self, parsed_data: Optional[List[Dict]] = None) -> List[Dict]:
        """
        Convert parsed data into RAG-ready chunks.

        Args:
            parsed_data: Optional pre-parsed data. If None, calls parse() first.

        Returns:
            List of chunk dictionaries with:
            - text: str (content for embedding)
            - metadata: dict (type-specific metadata for JSONB storage)
            - hash: str (MD5 hash for deduplication)
            - num_tokens: int (optional, token count)
            - num_chars: int (character count)

        Example:
            [
                {
                    "text": "SCENE 12: INT. CAFE - DAY\\n\\nAlice sits...",
                    "metadata": {"scene_number": 12, "heading": "INT. CAFE - DAY", "location": "CAFE"},
                    "hash": "a3f2c9d",
                    "num_chars": 450
                },
                ...
            ]
        """
        pass

    def extract_metadata(self) -> Dict:
        """
        Extract document-level metadata for documents.metadata JSONB field.

        Returns:
            Dictionary with doc-specific metadata

        Examples:
            Books:         {"publisher": "Penguin", "year": 1925, "genre": "fiction"}
            Scripts:       {"screenplay_by": "...", "runtime_minutes": 120}
            Conversations: {"participants": ["Alice", "Bob"], "duration_seconds": 3600}
            Tech Docs:     {"version": "2.0", "has_code": true, "has_diagrams": true}
        """
        return {}

    def extract_assets(self) -> List[Dict]:
        """
        Extract rich content (images, tables, code blocks, diagrams).

        Returns:
            List of asset dictionaries for content_assets table

        Example:
            [
                {
                    "asset_type": "code",
                    "content": "def function():\\n    pass",
                    "page_number": 15,
                    "metadata": {"language": "python", "lines": 2}
                },
                {
                    "asset_type": "image",
                    "file_path": "extracted_images/fig_3_2.png",
                    "page_number": 42,
                    "metadata": {"caption": "Figure 3.2", "width": 800, "height": 600}
                },
                ...
            ]
        """
        return []

    @staticmethod
    def simple_hash(text: str, length: int = 7) -> str:
        """
        Generate short MD5 hash for text (for chunk deduplication).

        Args:
            text: Text to hash
            length: Hash length (default 7 chars)

        Returns:
            Hex hash string
        """
        return hashlib.md5(text.encode("utf-8")).hexdigest()[:length]

    def validate_chunks(self, chunks: List[Dict]) -> bool:
        """
        Validate that chunks have required fields.

        Args:
            chunks: List of chunk dictionaries

        Returns:
            True if valid, raises ValueError otherwise
        """
        required_fields = {"text", "metadata"}

        for i, chunk in enumerate(chunks):
            missing = required_fields - set(chunk.keys())
            if missing:
                raise ValueError(
                    f"Chunk {i} missing required fields: {missing}. "
                    f"Got: {list(chunk.keys())}"
                )

            if not isinstance(chunk["text"], str):
                raise ValueError(f"Chunk {i} 'text' must be string, got {type(chunk['text'])}")

            if not isinstance(chunk["metadata"], dict):
                raise ValueError(f"Chunk {i} 'metadata' must be dict, got {type(chunk['metadata'])}")

        return True

    def __repr__(self):
        return f"{self.__class__.__name__}(file={self.file_path.name}, slug={self.slug}, type={self.doc_type.value})"
