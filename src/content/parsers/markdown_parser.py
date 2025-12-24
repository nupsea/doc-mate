"""
Markdown/Technical Documentation parser - Section-based chunking with code/table detection.

Supports:
- Markdown files (.md)
- PDFs with technical content (code, tables, diagrams)
- Section-based splitting (headings)
- Code block extraction
- Table detection (structure lost, content preserved)
"""

import re
import tiktoken
from typing import List, Dict, Optional

try:
    import fitz  # PyMuPDF
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False

from .base import DocumentParser, DocumentType


class MarkdownParser(DocumentParser):
    """
    Parse technical documentation with section-based structure.

    Philosophy:
    - Split by section headings (##, ###, or PDF sections)
    - Preserve code blocks intact
    - Detect tables and diagrams
    - Keep chunks aligned with semantic sections
    """

    def __init__(self, file_path: str, slug: str, split_pattern: str = None):
        """
        Initialize markdown/tech doc parser.

        Args:
            file_path: Path to .md or .pdf file
            slug: Document identifier
            split_pattern: Optional heading pattern (default: ## headings)
        """
        # Default: Match markdown headings (##, ###, etc.)
        self.default_pattern = r'^#{2,3}\s+(.+)$'
        super().__init__(file_path, slug, split_pattern=split_pattern or self.default_pattern)

        # Token encoder
        self.enc = tiktoken.get_encoding("cl100k_base")

        # Code block patterns
        self.code_fence_pattern = r'```(\w+)?\n(.*?)```'
        self.code_inline_pattern = r'`([^`]+)`'

    def _get_doc_type(self) -> DocumentType:
        return DocumentType.TECH_DOC

    def read_content(self) -> str:
        """Read content from markdown or PDF."""
        if self.file_path.suffix.lower() == '.pdf':
            return self._read_pdf()
        else:
            # Assume markdown or text - use safe encoding detection
            return self.safe_read_text()

    def _read_pdf(self) -> str:
        """Extract text from PDF using PyMuPDF."""
        if not PYMUPDF_AVAILABLE:
            raise ImportError(
                "PyMuPDF required for PDF parsing. Install: pip install pymupdf"
            )

        doc = fitz.open(self.file_path)
        text_parts = []

        for page in doc:
            text = page.get_text("text")
            if text:
                text_parts.append(text)

        doc.close()
        return "\n".join(text_parts)

    def parse(self) -> List[Dict]:
        """
        Parse technical doc into sections.

        Returns:
            List of section dictionaries:
            [
                {
                    "section": "3.2",
                    "heading": "API Design",
                    "text": "...",
                    "has_code": True,
                    "has_table": False
                },
                ...
            ]
        """
        content = self.read_content()

        # Detect if markdown or plain text
        is_markdown = self.file_path.suffix.lower() == '.md' or '```' in content

        if is_markdown:
            return self._parse_markdown(content)
        else:
            return self._parse_plain_text(content)

    def _parse_markdown(self, content: str) -> List[Dict]:
        """Parse markdown with heading-based sections."""
        lines = content.split('\n')
        sections = []
        current_section = None
        current_text = []
        section_num = 0

        for line in lines:
            # Check for heading (##, ###)
            heading_match = re.match(r'^(#{2,3})\s+(.+)$', line)

            if heading_match:
                # Save previous section
                if current_section:
                    current_section["text"] = '\n'.join(current_text).strip()
                    sections.append(current_section)
                    current_text = []

                # Start new section
                section_num += 1
                heading_level = len(heading_match.group(1))
                heading_text = heading_match.group(2).strip()

                current_section = {
                    "section": str(section_num),
                    "heading": heading_text,
                    "level": heading_level,
                    "text": "",
                    "has_code": False,
                    "has_table": False
                }
            else:
                # Accumulate section text
                if current_section:
                    current_text.append(line)

        # Don't forget last section
        if current_section:
            current_section["text"] = '\n'.join(current_text).strip()
            sections.append(current_section)

        # Detect code and tables in sections
        for section in sections:
            section["has_code"] = self._has_code(section["text"])
            section["has_table"] = self._has_table(section["text"])

        return sections

    def _parse_plain_text(self, content: str) -> List[Dict]:
        """Parse plain text (PDF) by detecting section patterns."""
        # For PDFs, we'll use simpler heuristics
        # Split on common section patterns or every N characters

        # Try to find numbered sections (1.2, 3.4, etc.)
        section_pattern = r'^(\d+\.?\d*)\s+(.+)$'
        lines = content.split('\n')

        sections = []
        current_section = None
        current_text = []
        section_num = 0

        for line in lines:
            # Check for section number
            section_match = re.match(section_pattern, line.strip())

            if section_match and len(line.strip()) < 100:  # Likely a heading
                # Save previous section
                if current_section:
                    current_section["text"] = '\n'.join(current_text).strip()
                    sections.append(current_section)
                    current_text = []

                # Start new section
                section_num += 1
                section_id = section_match.group(1)
                heading_text = section_match.group(2).strip()

                current_section = {
                    "section": section_id,
                    "heading": heading_text,
                    "text": "",
                    "has_code": False,
                    "has_table": False
                }
            else:
                # Accumulate text
                if current_section:
                    current_text.append(line)

        # Don't forget last section
        if current_section:
            current_section["text"] = '\n'.join(current_text).strip()
            sections.append(current_section)

        # If no sections found, treat as single section
        if not sections:
            sections = [{
                "section": "1",
                "heading": "Document",
                "text": content,
                "has_code": self._has_code(content),
                "has_table": self._has_table(content)
            }]
        else:
            # Detect code and tables
            for section in sections:
                section["has_code"] = self._has_code(section["text"])
                section["has_table"] = self._has_table(section["text"])

        return sections

    def chunk(self, parsed_data: Optional[List[Dict]] = None, max_tokens: int = 800, overlap: int = 50) -> List[Dict]:
        """
        Convert sections into RAG-ready chunks.

        Strategy:
        - Prefer full sections if they fit in max_tokens
        - Split large sections with overlap
        - Keep code blocks intact when possible

        Args:
            parsed_data: Optional pre-parsed sections
            max_tokens: Maximum tokens per chunk
            overlap: Token overlap between chunks

        Returns:
            List of chunk dictionaries
        """
        if parsed_data is None:
            parsed_data = self.parse()

        all_chunks = []
        section_num = 0

        for section in parsed_data:
            section_num += 1  # 1-indexed section number for ID format
            section_text = section["text"]
            section_tokens = len(self.enc.encode(section_text))
            chunk_in_section = 0

            if section_tokens <= max_tokens:
                # Section fits in one chunk
                chunk_in_section += 1
                chunk_hash = self.simple_hash(section_text)
                chunk_id = f"{self.slug}_{section_num:02d}_{chunk_in_section:03d}_{chunk_hash}"

                all_chunks.append({
                    "id": chunk_id,
                    "text": f"## {section['heading']}\n\n{section_text}",
                    "hash": chunk_hash,
                    "num_chars": len(section_text),
                    "num_tokens": section_tokens,
                    "metadata": {
                        "section": section["section"],
                        "heading": section["heading"],
                        "has_code": section.get("has_code", False),
                        "has_table": section.get("has_table", False)
                    }
                })
            else:
                # Split large section
                sub_chunks = self._chunk_text(section_text, max_tokens, overlap)

                for sub_chunk, sub_chunk_tokens in sub_chunks:
                    chunk_in_section += 1
                    chunk_hash = self.simple_hash(sub_chunk)
                    chunk_id = f"{self.slug}_{section_num:02d}_{chunk_in_section:03d}_{chunk_hash}"

                    all_chunks.append({
                        "id": chunk_id,
                        "text": sub_chunk,
                        "hash": chunk_hash,
                        "num_chars": len(sub_chunk),
                        "num_tokens": sub_chunk_tokens,  # Use pre-calculated tokens
                        "metadata": {
                            "section": section["section"],
                            "heading": section["heading"],
                            "has_code": section.get("has_code", False),
                            "has_table": section.get("has_table", False),
                            "is_partial": True
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

    def _has_code(self, text: str) -> bool:
        """Detect if text contains code blocks."""
        # Check for code fences
        if re.search(self.code_fence_pattern, text, re.DOTALL):
            return True

        # Check for common code keywords
        code_keywords = ['def ', 'class ', 'function ', 'SELECT ', 'CREATE ', 'INSERT ', 'import ', 'const ', 'let ', 'var ']
        return any(keyword in text for keyword in code_keywords)

    def _has_table(self, text: str) -> bool:
        """Detect if text contains tables (heuristic)."""
        # Check for markdown tables
        if re.search(r'\|.*\|', text):
            return True

        # Check for aligned columns (multiple spaces/tabs)
        lines = text.split('\n')
        table_lines = sum(1 for line in lines if line.count('  ') >= 3 or '\t' in line)

        return table_lines >= 3

    def extract_metadata(self) -> Dict:
        """
        Extract tech doc metadata.

        Returns:
            Metadata dictionary
        """
        sections = self.parse()

        return {
            "source_format": self.file_path.suffix,
            "num_sections": len(sections),
            "has_code": any(s.get("has_code") for s in sections),
            "has_tables": any(s.get("has_table") for s in sections),
            "section_headings": [s["heading"] for s in sections[:10]]  # First 10
        }

    def extract_assets(self) -> List[Dict]:
        """
        Extract code blocks from document.

        Returns:
            List of code block assets
        """
        content = self.read_content()
        assets = []

        # Extract code blocks
        for match in re.finditer(self.code_fence_pattern, content, re.DOTALL):
            language = match.group(1) or "unknown"
            code = match.group(2).strip()

            assets.append({
                "asset_type": "code",
                "content": code,
                "metadata": {
                    "language": language,
                    "lines": len(code.split('\n'))
                }
            })

        return assets
