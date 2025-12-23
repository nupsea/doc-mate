"""
Report parser - Section-based chunking for business reports, whitepapers, etc.

Supports:
- Executive summaries
- Section-based structure
- Tables and charts (content preserved, structure lost)
- Key findings and conclusions
"""

import re
import tiktoken
from pathlib import Path
from typing import List, Dict, Optional

try:
    import fitz  # PyMuPDF
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False

from .base import DocumentParser, DocumentType


class ReportParser(DocumentParser):
    """
    Parse business reports with section-based structure.

    Philosophy:
    - Identify common report sections (Executive Summary, Introduction, etc.)
    - Preserve table content (structure lost)
    - Extract key findings
    - Section-based chunking
    """

    def __init__(self, file_path: str, slug: str, split_pattern: str = None):
        """
        Initialize report parser.

        Args:
            file_path: Path to report file (.pdf, .txt, .md)
            slug: Document identifier
            split_pattern: Optional pattern for section detection
        """
        # Default: Common report section patterns
        self.default_pattern = r'^(Executive Summary|Introduction|Background|Methods?|Results?|Findings?|Analysis|Discussion|Conclusion|Recommendations?|Appendix)'
        super().__init__(file_path, slug, split_pattern=split_pattern or self.default_pattern)

        # Token encoder
        self.enc = tiktoken.get_encoding("cl100k_base")

    def _get_doc_type(self) -> DocumentType:
        return DocumentType.REPORT

    def read_content(self) -> str:
        """Read content from PDF or text file."""
        if self.file_path.suffix.lower() == '.pdf':
            return self._read_pdf()
        else:
            # Use safe encoding detection
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
        Parse report into sections.

        Returns:
            List of section dictionaries:
            [
                {
                    "section_type": "executive_summary",
                    "heading": "Executive Summary",
                    "text": "...",
                    "has_table": False
                },
                ...
            ]
        """
        content = self.read_content()
        lines = content.split('\n')

        sections = []
        current_section = None
        current_text = []

        # Common section patterns (case-insensitive)
        section_keywords = [
            'executive summary', 'introduction', 'background',
            'methodology', 'methods', 'approach',
            'results', 'findings', 'analysis',
            'discussion', 'conclusion', 'recommendations',
            'appendix', 'references'
        ]

        for line in lines:
            line_stripped = line.strip()

            # Check if line is a section heading
            is_section = False
            section_type = None

            for keyword in section_keywords:
                if keyword in line_stripped.lower() and len(line_stripped) < 100:
                    # Likely a section heading
                    is_section = True
                    section_type = keyword.replace(' ', '_')
                    break

            if is_section:
                # Save previous section
                if current_section and current_text:
                    current_section["text"] = '\n'.join(current_text).strip()
                    sections.append(current_section)
                    current_text = []

                # Start new section
                current_section = {
                    "section_type": section_type,
                    "heading": line_stripped,
                    "text": "",
                    "has_table": False
                }
            else:
                # Accumulate section text
                if current_section:
                    current_text.append(line)

        # Don't forget last section
        if current_section and current_text:
            current_section["text"] = '\n'.join(current_text).strip()
            sections.append(current_section)

        # If no sections found, treat as single document
        if not sections:
            sections = [{
                "section_type": "full_document",
                "heading": "Full Report",
                "text": content,
                "has_table": False
            }]

        # Detect tables in sections
        for section in sections:
            section["has_table"] = self._has_table(section["text"])

        return sections

    def chunk(self, parsed_data: Optional[List[Dict]] = None, max_tokens: int = 700, overlap: int = 50) -> List[Dict]:
        """
        Convert report sections into RAG-ready chunks.

        Strategy:
        - Keep sections intact if possible
        - Split large sections with overlap
        - Preserve section type in metadata

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
                chunk_text = f"{section['heading']}\n\n{section_text}"
                chunk_hash = self.simple_hash(chunk_text)

                # Use consistent ID format: {slug}_{section:02d}_{chunk:03d}_{hash}
                chunk_id = f"{self.slug}_{section_num:02d}_{chunk_in_section:03d}_{chunk_hash}"

                all_chunks.append({
                    "id": chunk_id,
                    "text": chunk_text,
                    "hash": chunk_hash,
                    "num_chars": len(chunk_text),
                    "num_tokens": section_tokens,
                    "metadata": {
                        "section_type": section["section_type"],
                        "heading": section["heading"],
                        "has_table": section.get("has_table", False)
                    }
                })
            else:
                # Split large section
                sub_chunks = self._chunk_text(section_text, max_tokens, overlap)

                for sub_chunk, sub_chunk_tokens in sub_chunks:
                    chunk_in_section += 1
                    chunk_hash = self.simple_hash(sub_chunk)

                    # Use consistent ID format: {slug}_{section:02d}_{chunk:03d}_{hash}
                    chunk_id = f"{self.slug}_{section_num:02d}_{chunk_in_section:03d}_{chunk_hash}"

                    all_chunks.append({
                        "id": chunk_id,
                        "text": sub_chunk,
                        "hash": chunk_hash,
                        "num_chars": len(sub_chunk),
                        "num_tokens": sub_chunk_tokens,  # Use pre-calculated tokens
                        "metadata": {
                            "section_type": section["section_type"],
                            "heading": section["heading"],
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

    def _has_table(self, text: str) -> bool:
        """Detect if text contains tables (heuristic)."""
        # Check for aligned columns (multiple spaces/tabs)
        lines = text.split('\n')
        table_lines = sum(1 for line in lines if line.count('  ') >= 3 or '\t' in line)

        # Check for numeric data patterns
        numeric_pattern = r'(\d+\s+){3,}'
        has_aligned_numbers = bool(re.search(numeric_pattern, text))

        return table_lines >= 3 or has_aligned_numbers

    def extract_metadata(self) -> Dict:
        """
        Extract report-level metadata.

        Returns:
            Metadata dictionary
        """
        sections = self.parse()

        # Identify section types present
        section_types = [s["section_type"] for s in sections]

        # Check for executive summary
        has_executive_summary = "executive_summary" in section_types

        return {
            "source_format": self.file_path.suffix,
            "num_sections": len(sections),
            "section_types": list(set(section_types)),
            "has_executive_summary": has_executive_summary,
            "has_tables": any(s.get("has_table") for s in sections)
        }
