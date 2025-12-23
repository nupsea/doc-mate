"""
Conversation parser - Turn-based chunking for chat logs, transcripts, meetings.

Supports formats:
- "Speaker: message" (Slack, Discord)
- "[00:00] Speaker: message" (with timestamps)
- "Speaker (HH:MM): message"
- WhatsApp exports
"""

import re
import tiktoken
from pathlib import Path
from typing import List, Dict, Optional, Tuple

try:
    import fitz  # PyMuPDF
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False

from .base import DocumentParser, DocumentType


class ConversationParser(DocumentParser):
    """
    Parse conversations/transcripts with turn-based chunking.

    Philosophy:
    - Extract speaker turns (minimal parsing)
    - Group turns into logical chunks (conversation context)
    - Preserve speaker and timestamp info
    - Let LLM understand conversation flow
    """

    def __init__(self, file_path: str, slug: str, split_pattern: str = None):
        """
        Initialize conversation parser.

        Args:
            file_path: Path to conversation file (.txt, .log, .chat, .pdf)
            slug: Document identifier
            split_pattern: Optional pattern for turn detection
        """
        super().__init__(file_path, slug, split_pattern=split_pattern)

        # Token encoder for chunking
        self.enc = tiktoken.get_encoding("cl100k_base")

        # Common conversation patterns
        self.turn_patterns = [
            # "[00:12:34] Alice: message"
            r'^\[(\d{2}:\d{2}:\d{2})\]\s*([^:]+):\s*(.+)$',
            # "Alice (14:30): message"
            r'^([^(]+)\s*\((\d{2}:\d{2})\):\s*(.+)$',
            # "Alice: message"
            r'^([^:]+):\s*(.+)$',
            # "Speaker> message" (some chat systems)
            r'^([^>]+)>\s*(.+)$',
        ]

    def _get_doc_type(self) -> DocumentType:
        return DocumentType.CONVERSATION

    def read_content(self) -> str:
        """Read conversation from text file or PDF."""
        if self.file_path.suffix.lower() == '.pdf':
            return self._read_pdf()
        else:
            # Use safe encoding detection for text files
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
        Parse conversation into speaker turns.

        Returns:
            List of turn dictionaries:
            [
                {"turn": 1, "speaker": "Alice", "timestamp": "00:12:34", "text": "..."},
                {"turn": 2, "speaker": "Bob", "timestamp": "00:12:45", "text": "..."},
                ...
            ]
        """
        content = self.read_content()
        lines = content.split('\n')

        turns = []
        turn_num = 0
        current_speaker = None
        current_text = []
        current_timestamp = None

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Try to match turn patterns
            matched = False
            for pattern in self.turn_patterns:
                match = re.match(pattern, line)
                if match:
                    # Save previous turn
                    if current_speaker and current_text:
                        turn_num += 1
                        turns.append({
                            "turn": turn_num,
                            "speaker": current_speaker.strip(),
                            "timestamp": current_timestamp,
                            "text": '\n'.join(current_text).strip()
                        })
                        current_text = []

                    # Parse new turn
                    groups = match.groups()
                    if len(groups) == 3:
                        # Has timestamp
                        if ':' in groups[0] and len(groups[0]) > 5:  # Timestamp first
                            current_timestamp = groups[0]
                            current_speaker = groups[1]
                            current_text = [groups[2]]
                        else:  # Speaker first
                            current_speaker = groups[0]
                            current_timestamp = groups[1]
                            current_text = [groups[2]]
                    elif len(groups) == 2:
                        # No timestamp
                        current_speaker = groups[0]
                        current_timestamp = None
                        current_text = [groups[1]]

                    matched = True
                    break

            # If no match, append to current turn
            if not matched and current_speaker:
                current_text.append(line)

        # Don't forget last turn
        if current_speaker and current_text:
            turn_num += 1
            turns.append({
                "turn": turn_num,
                "speaker": current_speaker.strip(),
                "timestamp": current_timestamp,
                "text": '\n'.join(current_text).strip()
            })

        return turns

    def chunk(self, parsed_data: Optional[List[Dict]] = None, max_tokens: int = 500, overlap_turns: int = 2) -> List[Dict]:
        """
        Group speaker turns into logical chunks with overlap.

        Args:
            parsed_data: Optional pre-parsed turns
            max_tokens: Maximum tokens per chunk
            overlap_turns: Number of turns to overlap between chunks

        Returns:
            List of chunk dictionaries
        """
        if parsed_data is None:
            parsed_data = self.parse()

        if not parsed_data:
            return []

        # Pre-calculate token counts (performance optimization)
        turn_token_counts = [len(self.enc.encode(turn["text"])) for turn in parsed_data]

        chunks = []
        chunk_index = 0
        position = 0

        while position < len(parsed_data):
            # Collect turns for this chunk
            chunk_turns = []
            chunk_tokens = 0
            current = position

            while current < len(parsed_data):
                turn_tokens = turn_token_counts[current]

                # Stop if adding this turn would exceed limit (and we already have some turns)
                if chunk_tokens + turn_tokens > max_tokens and chunk_turns:
                    break

                chunk_turns.append(parsed_data[current])
                chunk_tokens += turn_tokens
                current += 1

            # Edge case: single turn exceeds max_tokens
            if not chunk_turns:
                chunk_turns = [parsed_data[position]]
                chunk_tokens = turn_token_counts[position]
                current = position + 1

            # Create chunk
            chunk_text = self._build_chunk_text(chunk_turns)
            chunk_index += 1
            chunk_hash = self.simple_hash(chunk_text)
            speakers = list(set(t["speaker"] for t in chunk_turns))

            chunks.append({
                "id": f"{self.slug}_{chunk_index:02d}_001_{chunk_hash}",
                "text": chunk_text,
                "hash": chunk_hash,
                "num_chars": len(chunk_text),
                "num_tokens": chunk_tokens,
                "metadata": {
                    "turn_start": chunk_turns[0]["turn"],
                    "turn_end": chunk_turns[-1]["turn"],
                    "speakers": speakers,
                    "timestamp_start": chunk_turns[0].get("timestamp"),
                    "timestamp_end": chunk_turns[-1].get("timestamp"),
                    "num_turns": len(chunk_turns)
                }
            })

            # Next chunk: move forward with overlap
            # Always advance by at least 1 to guarantee termination
            position = max(position + 1, current - overlap_turns)

        print(f"[CHUNK] Completed! Created {len(chunks)} chunks from {len(parsed_data)} turns")
        return chunks

    def _build_chunk_text(self, turns: List[Dict]) -> str:
        """Build formatted chunk text from turns."""
        lines = []

        for turn in turns:
            # Format: "Alice (00:12:34): message"
            if turn.get("timestamp"):
                lines.append(f"{turn['speaker']} ({turn['timestamp']}): {turn['text']}")
            else:
                lines.append(f"{turn['speaker']}: {turn['text']}")

        return '\n'.join(lines)

    def extract_metadata(self) -> Dict:
        """
        Extract conversation-level metadata.

        Returns:
            Metadata dictionary with participants, duration, etc.
        """
        turns = self.parse()

        if not turns:
            return {}

        # Extract unique speakers
        speakers = list(set(t["speaker"] for t in turns))

        # Extract timestamps if available
        timestamps = [t.get("timestamp") for t in turns if t.get("timestamp")]

        # Calculate duration if timestamps available
        duration_seconds = None
        if len(timestamps) >= 2:
            try:
                start_seconds = self._parse_timestamp(timestamps[0])
                end_seconds = self._parse_timestamp(timestamps[-1])
                duration_seconds = end_seconds - start_seconds
            except:
                pass

        return {
            "participants": speakers,
            "num_participants": len(speakers),
            "num_turns": len(turns),
            "has_timestamps": bool(timestamps),
            "duration_seconds": duration_seconds
        }

    def _parse_timestamp(self, timestamp: str) -> int:
        """Convert timestamp string to seconds."""
        # Parse formats like "00:12:34" or "12:34"
        parts = timestamp.split(':')
        if len(parts) == 3:
            h, m, s = map(int, parts)
            return h * 3600 + m * 60 + s
        elif len(parts) == 2:
            m, s = map(int, parts)
            return m * 60 + s
        return 0
