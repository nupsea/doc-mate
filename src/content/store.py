import os
import json
import psycopg2
from psycopg2.extras import execute_values

from dotenv import load_dotenv

load_dotenv()  # reads .env

DB_CONFIG = {
    "dbname": os.getenv("PG_DB", "booksdb"),
    "user": os.getenv("PG_USER", "bookuser"),
    "password": os.getenv("PG_PASS", "bookpass"),
    "host": os.getenv("PG_HOST", "localhost"),
    "port": os.getenv("PG_PORT", 5432),
}


class PgresStore:

    def __init__(self, conn=None) -> None:
        self.conn = conn or psycopg2.connect(**DB_CONFIG)

    def _resolve_doc_id(self, doc_identifier: int | str) -> int | None:
        """
        Resolve doc_id from either:
        - doc_id (int): returns as-is
        - slug (str): looks up doc_id from documents table
        - title (str): fallback to case-insensitive title match
        Returns None if not found.
        """
        if isinstance(doc_identifier, int):
            return doc_identifier

        # Try exact slug match first
        with self.conn.cursor() as cur:
            cur.execute("SELECT doc_id FROM documents WHERE slug = %s", (doc_identifier,))
            row = cur.fetchone()
            if row:
                return row[0]

            # Fallback: try case-insensitive title match
            cur.execute("SELECT doc_id FROM documents WHERE LOWER(title) = LOWER(%s)", (doc_identifier,))
            row = cur.fetchone()
            return row[0] if row else None

    def document_exists(self, slug: str) -> bool:
        """Check if a document with given slug exists."""
        with self.conn.cursor() as cur:
            cur.execute("SELECT 1 FROM documents WHERE slug = %s", (slug,))
            return cur.fetchone() is not None

    def summaries_exist(self, doc_identifier: int | str) -> bool:
        """Check if summaries exist for a document."""
        doc_id = self._resolve_doc_id(doc_identifier)
        if not doc_id:
            return False

        with self.conn.cursor() as cur:
            # Check if document summary exists
            cur.execute("SELECT 1 FROM document_summaries WHERE doc_id = %s", (doc_id,))
            return cur.fetchone() is not None

    def delete_document(self, doc_identifier: int | str) -> bool:
        """
        Delete a document and all related data (summaries).
        Foreign key CASCADE will automatically delete chapter_summaries and document_summaries.
        Returns True if document was deleted, False if not found.
        """
        doc_id = self._resolve_doc_id(doc_identifier)
        if not doc_id:
            return False

        with self.conn.cursor() as cur:
            cur.execute("DELETE FROM documents WHERE doc_id = %s", (doc_id,))
            deleted = cur.rowcount > 0
        self.conn.commit()
        return deleted

    def store_document_metadata(
        self,
        slug: str,
        title: str,
        author: str = None,
        num_chunks: int = None,
        num_chars: int = None,
    ) -> int:
        """
        Insert or update document metadata. Returns doc_id.

        DEPRECATED: Use store_document() instead for multi-format support.
        Kept for backward compatibility.
        """
        return self.store_document(
            slug=slug,
            title=title,
            doc_type='book',
            author=author,
            num_chunks=num_chunks,
            num_chars=num_chars
        )

    def _sanitize_metadata(self, metadata: dict) -> dict:
        """Remove null bytes from metadata that PostgreSQL JSONB can't handle."""
        if not metadata:
            return {}

        def clean_value(val):
            if isinstance(val, str):
                return val.replace('\x00', '')
            elif isinstance(val, dict):
                return {k: clean_value(v) for k, v in val.items()}
            elif isinstance(val, list):
                return [clean_value(item) for item in val]
            return val

        return clean_value(metadata)

    def store_document(
        self,
        slug: str,
        title: str,
        doc_type: str = 'book',
        author: str = None,
        num_chunks: int = None,
        num_chars: int = None,
        metadata: dict = None,
        is_ephemeral: bool = False,
    ) -> int:
        """
        Store any document type (books, scripts, conversations, etc).

        Args:
            slug: Unique identifier for document
            title: Document title
            doc_type: 'book', 'script', 'conversation', 'tech_doc', 'report'
            author: Document author (optional)
            num_chunks: Number of chunks created
            num_chars: Total character count
            metadata: Type-specific metadata (stored as JSONB)
            is_ephemeral: Whether document is ephemeral (default False)

        Returns:
            doc_id (document ID)
        """
        # Sanitize metadata to remove null bytes
        clean_metadata = self._sanitize_metadata(metadata)

        with self.conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO documents (slug, title, author, num_chunks, num_chars, doc_type, metadata, is_ephemeral)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (slug) DO UPDATE
                SET title = excluded.title,
                    author = excluded.author,
                    num_chunks = excluded.num_chunks,
                    num_chars = excluded.num_chars,
                    doc_type = excluded.doc_type,
                    metadata = excluded.metadata,
                    is_ephemeral = excluded.is_ephemeral
                RETURNING doc_id
                """,
                (slug, title, author, num_chunks, num_chars, doc_type,
                 json.dumps(clean_metadata), is_ephemeral),
            )
            doc_id = cur.fetchone()[0]
        self.conn.commit()
        return doc_id

    def store_summaries(
        self, doc_identifier: int | str, chapter_summaries: list, document_summary: str
    ):
        """
        Store chapter and document summaries.
        doc_identifier can be either doc_id (int) or slug (str).
        """
        doc_id = self._resolve_doc_id(doc_identifier)
        if not doc_id:
            raise ValueError(f"Document not found: {doc_identifier}")

        with self.conn.cursor() as cur:
            # Insert chapters
            rows = [(doc_id, c["chapter_id"], c["summary"]) for c in chapter_summaries]
            execute_values(
                cur,
                """
                INSERT INTO chapter_summaries (doc_id, chapter_id, summary)
                VALUES %s
                ON CONFLICT (doc_id, chapter_id) DO UPDATE SET summary = excluded.summary
            """,
                rows,
            )

            # Insert document summary
            cur.execute(
                """
                INSERT INTO document_summaries (doc_id, summary)
                VALUES (%s, %s)
                ON CONFLICT (doc_id) DO UPDATE SET summary = excluded.summary
            """,
                (doc_id, document_summary),
            )

        self.conn.commit()

    def get_chapter_summary(
        self, doc_identifier: int | str, chapter_id: int
    ) -> str | None:
        """
        Fetch one chapter summary from DB.
        doc_identifier can be either doc_id (int) or slug (str).
        """
        doc_id = self._resolve_doc_id(doc_identifier)
        if not doc_id:
            return None

        with self.conn.cursor() as cur:
            cur.execute(
                """
                SELECT summary
                FROM chapter_summaries
                WHERE doc_id = %s AND chapter_id = %s
                """,
                (doc_id, chapter_id),
            )
            row = cur.fetchone()
        return row[0] if row else None

    def get_all_chapter_summaries(
        self, doc_identifier: int | str
    ) -> list[tuple[int, str]]:
        """
        Fetch all chapter summaries for a document, ordered by chapter_id.
        doc_identifier can be either doc_id (int) or slug (str).
        """
        doc_id = self._resolve_doc_id(doc_identifier)
        if not doc_id:
            return []

        with self.conn.cursor() as cur:
            cur.execute(
                """
                SELECT chapter_id, summary
                FROM chapter_summaries
                WHERE doc_id = %s
                ORDER BY chapter_id
                """,
                (doc_id,),
            )
            rows = cur.fetchall()
        return rows

    def get_document_summary(self, doc_identifier: int | str) -> str | None:
        """
        Fetch the overall document summary.
        doc_identifier can be either doc_id (int) or slug (str).
        """
        doc_id = self._resolve_doc_id(doc_identifier)
        if not doc_id:
            return None

        with self.conn.cursor() as cur:
            cur.execute(
                """
                SELECT summary
                FROM document_summaries
                WHERE doc_id = %s
                """,
                (doc_id,),
            )
            row = cur.fetchone()
        return row[0] if row else None

    def store_bm25_index(self, term_freqs: list[tuple[str, str, int]], doc_lens: list[tuple[str, int]]):
        """
        Store BM25 term frequencies and document lengths in bulk.
        term_freqs: list of (term, chunk_id, frequency)
        doc_lens: list of (chunk_id, doc_len)
        """
        with self.conn.cursor() as cur:
            # Store term frequencies
            execute_values(
                cur,
                """
                INSERT INTO bm25_index (term, chunk_id, frequency)
                VALUES %s
                ON CONFLICT (term, chunk_id) DO UPDATE SET frequency = excluded.frequency
                """,
                term_freqs,
            )

            # Store document lengths
            execute_values(
                cur,
                """
                INSERT INTO bm25_doc_lens (chunk_id, doc_len)
                VALUES %s
                ON CONFLICT (chunk_id) DO UPDATE SET doc_len = excluded.doc_len
                """,
                doc_lens,
            )
        self.conn.commit()

    def get_bm25_stats(self, terms: list[str], doc_slug: str = None):
        """
        Retrieve statistics needed for BM25 scoring.
        Returns:
            - df: {term: doc_frequency}
            - tf: {chunk_id: {term: frequency}}
            - doc_lens: {chunk_id: length}
            - N: total number of documents
        """
        with self.conn.cursor() as cur:
            # 1. Total document count (N)
            if doc_slug:
                cur.execute("SELECT COUNT(*) FROM bm25_doc_lens WHERE chunk_id LIKE %s", (f"{doc_slug}_%",))
            else:
                cur.execute("SELECT COUNT(*) FROM bm25_doc_lens")
            N = cur.fetchone()[0]

            # 2. Document frequencies (df) for the requested terms
            df = {}
            if terms:
                if doc_slug:
                    cur.execute(
                        "SELECT term, COUNT(chunk_id) FROM bm25_index WHERE term = ANY(%s) AND chunk_id LIKE %s GROUP BY term",
                        (terms, f"{doc_slug}_%"),
                    )
                else:
                    cur.execute(
                        "SELECT term, COUNT(chunk_id) FROM bm25_index WHERE term = ANY(%s) GROUP BY term",
                        (terms,),
                    )
                df = {row[0]: row[1] for row in cur.fetchall()}

            # 3. Term frequencies (tf) and Document lengths (doc_lens)
            tf = {}
            doc_lens = {}
            if terms:
                if doc_slug:
                    cur.execute(
                        """
                        SELECT bi.chunk_id, bi.term, bi.frequency, dl.doc_len 
                        FROM bm25_index bi
                        JOIN bm25_doc_lens dl ON bi.chunk_id = dl.chunk_id
                        WHERE bi.term = ANY(%s) AND bi.chunk_id LIKE %s
                        """,
                        (terms, f"{doc_slug}_%"),
                    )
                else:
                    cur.execute(
                        """
                        SELECT bi.chunk_id, bi.term, bi.frequency, dl.doc_len 
                        FROM bm25_index bi
                        JOIN bm25_doc_lens dl ON bi.chunk_id = dl.chunk_id
                        WHERE bi.term = ANY(%s)
                        """,
                        (terms,),
                    )
                
                for chunk_id, term, frequency, doc_len in cur.fetchall():
                    if chunk_id not in tf:
                        tf[chunk_id] = {}
                    tf[chunk_id][term] = frequency
                    doc_lens[chunk_id] = doc_len

        return {"df": df, "tf": tf, "doc_lens": doc_lens, "N": N}