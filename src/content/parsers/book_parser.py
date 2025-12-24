"""
Book parser - Chapter-based chunking for novels, textbooks, etc.

Supports:
- Gutenberg text files
- PDF books
- Chapter-based splitting
- Token-based chunking with overlap
"""

import re
import tiktoken
from typing import List, Dict, Optional

try:
    import fitz  # PyMuPDF
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False

try:
    import pdfplumber
    PDFPLUMBER_AVAILABLE = True
except ImportError:
    PDFPLUMBER_AVAILABLE = False

from .base import DocumentParser, DocumentType


class BookParser(DocumentParser):
    """
    Parse books with chapter-based structure.

    Philosophy:
    - Split by chapters (configurable pattern)
    - Chunk within chapters with overlap
    - Preserve chapter context in metadata
    """

    def __init__(self, file_path: str, slug: str, split_pattern: str = None):
        """
        Initialize book parser.

        Args:
            file_path: Path to book file (.txt or .pdf)
            slug: Document identifier
            split_pattern: Regex for chapter detection (default: CHAPTER patterns)
        """
        # Default chapter patterns
        self.default_pattern = r"^(?:CHAPTER [IVXLCDM]+\.?|Chapter \d+)\s*\n"
        super().__init__(file_path, slug, split_pattern=split_pattern or self.default_pattern)

        # Token encoder
        self.enc = tiktoken.get_encoding("cl100k_base")

    def _get_doc_type(self) -> DocumentType:
        return DocumentType.BOOK

    def read_content(self) -> str:
        """Read content from text or PDF file."""
        if self.file_path.suffix.lower() == '.pdf':
            return self._read_pdf()
        else:
            # Assume text file - use safe encoding detection
            text = self.safe_read_text()
            return self._strip_gutenberg(text)

    def _read_pdf(self) -> str:
        """Extract text from PDF using available libraries."""
        if PYMUPDF_AVAILABLE:
            return self._read_pdf_pymupdf()
        elif PDFPLUMBER_AVAILABLE:
            return self._read_pdf_pdfplumber()
        else:
            raise ImportError(
                "No PDF library available. Install pymupdf or pdfplumber: "
                "pip install pymupdf  OR  pip install pdfplumber"
            )

    def _read_pdf_pymupdf(self) -> str:
        """Extract text using PyMuPDF (faster, better quality)."""
        doc = fitz.open(self.file_path)
        text_parts = []

        for page in doc:
            text = page.get_text("text")
            if text:
                text_parts.append(text)

        doc.close()
        return "\n".join(text_parts)

    def _read_pdf_pdfplumber(self) -> str:
        """Extract text using pdfplumber (fallback)."""
        text_parts = []

        with pdfplumber.open(self.file_path) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    text_parts.append(text)

        return "\n".join(text_parts)

    def _strip_gutenberg(self, text: str) -> str:
        """Remove Gutenberg header/footer boilerplate."""
        start_match = re.search(r"\*\*\* START OF.*\*\*\*", text)
        end_match = re.search(r"\*\*\* END OF.*\*\*\*", text)

        if start_match and end_match:
            return text[start_match.end() : end_match.start()].strip()
        return text

    def parse(self) -> List[Dict]:
        """
        Parse book into chapters.

        Returns:
            List of chapter dictionaries:
            [
                {"chapter": 1, "text": "...", "title": "Introduction"},
                {"chapter": 2, "text": "...", "title": "Methods"},
                ...
            ]
        """
        content = self.read_content()

        # Split by chapters
        parts = re.split(self.split_pattern, content, flags=re.IGNORECASE | re.MULTILINE)

        chapters = []
        chapter_num = 0

        for part in parts:
            if not part.strip():
                continue

            chapter_num += 1

            # Try to extract chapter title (first line)
            lines = part.strip().split('\n', 1)
            title = lines[0][:100] if lines else f"Chapter {chapter_num}"

            chapters.append({
                "chapter": chapter_num,
                "title": title,
                "text": part.strip()
            })

        return chapters

    def chunk(self, parsed_data: Optional[List[Dict]] = None, max_tokens: int = 500, overlap: int = 100) -> List[Dict]:
        """
        Convert chapters into RAG-ready chunks with overlap.

        Args:
            parsed_data: Optional pre-parsed chapters
            max_tokens: Maximum tokens per chunk
            overlap: Token overlap between chunks

        Returns:
            List of chunk dictionaries
        """
        if parsed_data is None:
            parsed_data = self.parse()

        all_chunks = []
        chunk_index = 0

        for chapter in parsed_data:
            chapter_chunks = self._chunk_text(
                chapter["text"],
                max_tokens=max_tokens,
                overlap=overlap
            )

            for sub_chunk, sub_chunk_tokens in chapter_chunks:
                chunk_index += 1
                chunk_hash = self.simple_hash(sub_chunk)

                all_chunks.append({
                    "text": sub_chunk,
                    "hash": chunk_hash,
                    "num_chars": len(sub_chunk),
                    "num_tokens": sub_chunk_tokens,  # Use pre-calculated tokens
                    "metadata": {
                        "chapter": chapter["chapter"],
                        "chapter_title": chapter["title"],
                        "chunk_index": chunk_index
                    }
                })

        return all_chunks

    def _chunk_text(self, text: str, max_tokens: int, overlap: int) -> List[tuple]:
        """
        Split text into token-based chunks with overlap.

        Returns:
            List of (chunk_text, token_count) tuples
        """
        tokens = self.enc.encode(text)
        chunks = []

        for i in range(0, len(tokens), max_tokens - overlap):
            chunk_tokens = tokens[i : i + max_tokens]
            chunk_text = self.enc.decode(chunk_tokens)
            chunks.append((chunk_text, len(chunk_tokens)))

        return chunks

    def extract_metadata(self) -> Dict:
        """
        Extract book-level metadata.

        Returns:
            Metadata dictionary
        """
        content = self.read_content()

        return {
            "source_format": self.file_path.suffix,
            "total_chars": len(content),
            "total_words": len(content.split()),
            "split_pattern": self.split_pattern
        }
