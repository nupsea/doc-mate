"""
Re-ingest Gita with proper chapter splitting

The Gita uses chapter TITLES as markers, not "CHAPTER I" etc.
Example: "THE DISTRESS OF ARJUNA", "THE BOOK OF DOCTRINES", etc.
"""
import asyncio
import sys
import re
import hashlib
import tiktoken
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.content.store import PgresStore
from src.search.vec import SemanticRetriever


GITA_CHAPTERS = [
    "THE DISTRESS OF ARJUNA",
    "THE BOOK OF DOCTRINES",
    "VIRTUE IN WORK",
    "THE RELIGION OF KNOWLEDGE",
    "RELIGION OF RENOUNCING WORKS",
    "RELIGION BY SELF-RESTRAINT",
    "RELIGION BY DISCERNMENT",
    "RELIGION BY SERVICE OF THE SUPREME",
    "RELIGION BY THE KINGLY KNOWLEDGE AND THE KINGLY MYSTERY",
    "RELIGION BY THE HEAVENLY PERFECTIONS",
    "THE MANIFESTING OF THE ONE AND MANIFOLD",
    "RELIGION OF FAITH",
    "RELIGION BY SEPARATION OF MATTER AND SPIRIT",
    "RELIGION BY SEPARATION FROM THE QUALITIES",
    "RELIGION BY ATTAINING THE SUPREME",
    "THE SEPARATENESS OF THE DIVINE AND UNDIVINE",
    "RELIGION BY THE THREEFOLD FAITH",
    "RELIGION BY DELIVERANCE AND RENUNCIATION"
]


def strip_gutenberg(text: str) -> str:
    """Remove Gutenberg header/footer"""
    start_match = re.search(r'\*\*\* START OF.*\*\*\*', text)
    end_match = re.search(r'\*\*\* END OF.*\*\*\*', text)
    if start_match and end_match:
        return text[start_match.end():end_match.start()].strip()
    return text


def create_gita_pattern():
    """Create regex pattern to match Gita chapter endings"""
    # The Gita has chapter endings like "HERE ENDETH CHAPTER I. OF THE BHAGAVAD-GITA,"
    # We'll split on these markers
    pattern = r'^\s*HERE ENDETH CHAPTER [IVXLCDM]+\. OF THE.*$'
    return pattern


def parse_gita_chapters(file_path: str):
    """Parse Gita into chapters"""
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()

    # Remove Gutenberg boilerplate
    content = strip_gutenberg(content)

    # Split by chapter endings
    pattern = create_gita_pattern()
    parts = re.split(pattern, content, flags=re.MULTILINE)

    chapters = []
    chapter_num = 0

    for part in parts:
        part_stripped = part.strip()
        if not part_stripped or len(part_stripped) < 100:  # Skip empty or tiny fragments
            continue

        chapter_num += 1

        # Try to extract title from first lines (often "CHAPTER I" or section heading)
        lines = part_stripped.split('\n')
        title = f"Chapter {chapter_num}"  # Default
        for line in lines[:10]:  # Check first 10 lines
            line_stripped = line.strip()
            if line_stripped and len(line_stripped) > 5 and len(line_stripped) < 100:
                # Look for lines that might be titles (all caps or title case, not too long)
                if line_stripped.upper() == line_stripped or line_stripped.istitle():
                    title = line_stripped[:80]
                    break

        chapters.append({
            "chapter": chapter_num,
            "title": title,
            "text": part_stripped
        })

    return chapters


def chunk_text(text: str, max_tokens: int = 500, overlap: int = 100):
    """Split text into token-based chunks"""
    enc = tiktoken.get_encoding("cl100k_base")
    tokens = enc.encode(text)
    chunks = []

    for i in range(0, len(tokens), max_tokens - overlap):
        chunk_tokens = tokens[i:i + max_tokens]
        chunk_text = enc.decode(chunk_tokens)
        chunks.append((chunk_text, len(chunk_tokens)))

    return chunks


def create_chunks(chapters):
    """Convert chapters to RAG chunks"""
    all_chunks = []
    chunk_index = 0

    for chapter in chapters:
        chapter_chunks = chunk_text(chapter["text"])

        for sub_chunk, sub_chunk_tokens in chapter_chunks:
            chunk_index += 1
            chunk_hash = hashlib.md5(sub_chunk.encode()).hexdigest()[:8]

            all_chunks.append({
                "id": f"gita_{chapter['chapter']:02d}_{chunk_index:03d}_{chunk_hash}",
                "text": sub_chunk,
                "metadata": {
                    "chapter": chapter["chapter"],
                    "chapter_title": chapter["title"],
                    "chunk_index": chunk_index
                }
            })

    return all_chunks


async def main():
    print("="*80)
    print("RE-INGESTING GITA WITH PROPER CHAPTER SPLITTING")
    print("="*80)

    gita_path = Path("/Users/sethurama/DEV/LM/doc-mate/DATA/the_gita.txt")

    if not gita_path.exists():
        print(f"ERROR: Gita file not found at {gita_path}")
        return

    print("\nParsing chapters...")
    chapters = parse_gita_chapters(str(gita_path))
    print(f"✓ Found {len(chapters)} chapters")

    for i, ch in enumerate(chapters[:5]):
        print(f"  Chapter {i+1}: {ch['title'][:60]}... ({len(ch['text'])} chars)")
    if len(chapters) > 5:
        print(f"  ... and {len(chapters)-5} more chapters")

    print("\nChunking content...")
    chunks = create_chunks(chapters)
    print(f"✓ Created {len(chunks)} chunks")

    # Show chunk distribution by chapter
    from collections import Counter
    chapter_counts = Counter(chunk['metadata']['chapter'] for chunk in chunks)
    print("\nChunks per chapter:")
    for ch_num in sorted(chapter_counts.keys())[:10]:
        print(f"  Chapter {ch_num}: {chapter_counts[ch_num]} chunks")
    if len(chapter_counts) > 10:
        print(f"  ... and {len(chapter_counts)-10} more chapters")

    # Delete existing Gita data
    print("\nDeleting existing Gita data from database...")
    store = PgresStore()

    # Delete from vector store
    print("Deleting from Qdrant...")
    retriever = SemanticRetriever()

    from qdrant_client.models import Filter, FieldCondition, MatchText
    retriever.qdrant.delete(
        collection_name="book_chunks",
        points_selector=Filter(
            must=[FieldCondition(key="id", match=MatchText(text="gita"))]
        )
    )
    print("✓ Deleted from Qdrant")

    # Now ingest new data
    print("\nIngesting new Gita data...")

    # Insert chunks into vector store
    print("Building vector index...")
    retriever.build_index(chunks)
    print("✓ Vector index built")

    # Update metadata in database if needed
    print("Updating database metadata...")
    with store.conn.cursor() as cur:
        cur.execute(
            "UPDATE books SET title = %s, author = %s WHERE slug = %s",
            ("The Gita", "Unknown", "gita")
        )
    store.conn.commit()
    print("✓ Database updated")

    print("\n" + "="*80)
    print("RE-INGESTION COMPLETE!")
    print("="*80)
    print(f"Chapters: {len(chapters)}")
    print(f"Chunks: {len(chunks)}")
    print("="*80)


if __name__ == "__main__":
    asyncio.run(main())
