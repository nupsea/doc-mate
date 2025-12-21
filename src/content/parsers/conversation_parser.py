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
            file_path: Path to conversation file (.txt, .log, .chat)
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
        """Read conversation from text file."""
        return self.file_path.read_text(encoding='utf-8')

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

        chunks = []
        chunk_index = 0

        i = 0
        while i < len(parsed_data):
            # Collect turns until we hit max_tokens
            chunk_turns = []
            chunk_tokens = 0

            while i < len(parsed_data):
                turn = parsed_data[i]
                turn_tokens = len(self.enc.encode(turn["text"]))

                if chunk_tokens + turn_tokens > max_tokens and chunk_turns:
                    # Chunk is full
                    break

                chunk_turns.append(turn)
                chunk_tokens += turn_tokens
                i += 1

            if not chunk_turns:
                # Single turn exceeds max_tokens, include it anyway
                chunk_turns = [parsed_data[i]]
                i += 1

            # Build chunk text
            chunk_text = self._build_chunk_text(chunk_turns)
            chunk_index += 1
            chunk_hash = self.simple_hash(chunk_text)

            # Extract speakers
            speakers = list(set(t["speaker"] for t in chunk_turns))

            # Create chunk ID using consistent format: {slug}_{section:02d}_{chunk:03d}_{hash}
            # For conversations, each chunk = 1 section, chunk number is always 001
            chunk_id = f"{self.slug}_{chunk_index:02d}_001_{chunk_hash}"

            chunks.append({
                "id": chunk_id,
                "text": chunk_text,
                "hash": chunk_hash,
                "num_chars": len(chunk_text),
                "num_tokens": len(self.enc.encode(chunk_text)),
                "metadata": {
                    "turn_start": chunk_turns[0]["turn"],
                    "turn_end": chunk_turns[-1]["turn"],
                    "speakers": speakers,
                    "timestamp_start": chunk_turns[0].get("timestamp"),
                    "timestamp_end": chunk_turns[-1].get("timestamp"),
                    "num_turns": len(chunk_turns)
                }
            })

            # Backtrack for overlap
            i -= overlap_turns

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
