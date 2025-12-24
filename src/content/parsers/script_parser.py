"""
Script Parser - Minimal, bulletproof approach for RAG.

Philosophy: Extract only what's 100% reliable. Let LLM understand everything else.
"""

import re
import tiktoken
from pathlib import Path
from typing import List, Dict, Optional
import fitz  # PyMuPDF

from .base import DocumentParser, DocumentType


class ScriptParser(DocumentParser):
    """
    Parse movie scripts into scene-based chunks.

    MINIMAL APPROACH - Extracts only bulletproof data:
    1. Scene boundaries (INT./EXT. pattern detection)
    2. Full heading text (unparsed - LLM will understand)
    3. Scene content (everything between headings)

    Does NOT parse:
    - Location (LLM reads from heading)
    - Time (LLM reads from heading)
    - INT/EXT distinction (LLM reads from heading)
    - Characters (LLM reads from dialogue)
    - Action vs dialogue (LLM understands naturally)

    Why? Zero edge cases, maximum reliability.
    """

    def __init__(self, file_path: str, slug: str, split_pattern: str = None):
        """
        Initialize script parser.

        Args:
            file_path: Path to script file (PDF or text)
            slug: Document identifier
            split_pattern: Not used (scenes are natural boundaries)
        """
        super().__init__(file_path, slug)

        # Token encoder for num_tokens
        self.enc = tiktoken.get_encoding("cl100k_base")

        # Minimal pattern: just detect scene headings
        # Matches: INT., EXT., INT/EXT., I/E., INT, EXT (any variation)
        self.scene_pattern = re.compile(
            r'^(INT\.|EXT\.|INT/EXT\.|I/E\.|INT |EXT )(.*)$',
            re.IGNORECASE
        )

    def _get_doc_type(self) -> DocumentType:
        """Return document type for script."""
        return DocumentType.SCRIPT

    def read_content(self) -> str:
        """Read script from PDF or text file."""
        file_path = Path(self.file_path)

        if file_path.suffix.lower() == '.pdf':
            return self._read_pdf()
        else:
            # Use safe encoding detection
            return self.safe_read_text()

    def _read_pdf(self) -> str:
        """Extract text from PDF screenplay."""
        doc = fitz.open(self.file_path)
        full_text = []

        for page in doc:
            text = page.get_text("text")
            full_text.append(text)

        doc.close()
        return "\n".join(full_text)

    def parse(self) -> List[Dict]:
        """
        Parse screenplay into scenes.

        Returns:
            List of scene dictionaries with ONLY:
            - scene_number: int (we generate this)
            - heading: str (full heading text, unparsed)
            - text: str (full scene content)
        """
        script_text = self.read_content()
        lines = script_text.split('\n')
        scenes = []
        current_scene = None
        current_lines = []

        for line in lines:
            stripped = line.strip()

            # Check for scene heading
            if self.scene_pattern.match(stripped):
                # Save previous scene
                if current_scene and current_lines:
                    current_scene['text'] = '\n'.join(current_lines).strip()
                    scenes.append(current_scene)
                    current_lines = []

                # Start new scene
                current_scene = {
                    'scene_number': len(scenes) + 1,
                    'heading': stripped,  # Keep full heading unparsed
                    'text': ''
                }
                continue

            # Collect all lines for current scene
            if current_scene:
                current_lines.append(line)

        # Don't forget last scene
        if current_scene and current_lines:
            current_scene['text'] = '\n'.join(current_lines).strip()
            scenes.append(current_scene)

        return scenes

    def chunk(self, parsed_data: Optional[List[Dict]] = None) -> List[Dict]:
        """
        Convert scenes into RAG-ready chunks.

        Each chunk: heading + scene text. That's it.
        LLM interprets everything else.

        Args:
            parsed_data: Optional pre-parsed scenes

        Returns:
            List of chunk dictionaries with:
            - text: str (heading + content for embedding)
            - metadata: dict (minimal metadata)
            - hash: str (MD5 hash for deduplication)
            - num_chars: int
        """
        if parsed_data is None:
            parsed_data = self.parse()

        chunks = []

        for scene in parsed_data:
            # Minimal chunk: just heading + content
            parts = [
                f"SCENE {scene['scene_number']}: {scene['heading']}",
                "",
                scene['text']
            ]

            chunk_text = "\n".join(parts)
            chunk_hash = self.simple_hash(chunk_text)
            chunk_tokens = len(self.enc.encode(chunk_text))  # Calculate once

            # Use consistent ID format: {slug}_{scene:02d}_{chunk:03d}_{hash}
            # Each scene = 1 chunk, so chunk number is always 001
            chunk_id = f"{self.slug}_{scene['scene_number']:02d}_001_{chunk_hash}"

            # Minimal metadata
            chunks.append({
                'id': chunk_id,
                'text': chunk_text,
                'hash': chunk_hash,
                'num_chars': len(chunk_text),
                'num_tokens': chunk_tokens,
                'metadata': {
                    'scene_number': scene['scene_number'],
                    'heading': scene['heading']  # Keep for filtering/display
                }
            })

        return chunks

    def extract_metadata(self) -> Dict:
        """Extract script-level metadata."""
        scenes = self.parse()

        return {
            "source_format": self.file_path.suffix,
            "num_scenes": len(scenes),
            "scene_headings": [s["heading"] for s in scenes[:10]]  # First 10 scenes
        }
