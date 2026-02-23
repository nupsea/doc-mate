import json
import contextlib
from typing import Any
from psycopg2.extras import execute_values
from dotenv import load_dotenv
from src.content.db import DatabaseManager

load_dotenv()  # reads .env

class PgresStore:
    """
    Postgres storage using connection pooling.
    Note: Methods here are synchronous. Use @async_db_task decorator or run_in_executor
    when calling from async contexts (agent graph).
    """

    def __init__(self, conn=None) -> None:
        # We no longer hold a persistent self.conn unless injected
        # Methods will request a connection from the pool on demand
        self._injected_conn = conn

    def _get_conn(self):
        """Helper to yield a connection context manager."""
        if self._injected_conn:
            # If a connection was injected (e.g. for testing), use it directly
            # We wrap it in a null context manager so 'with' works
            return contextlib.nullcontext(self._injected_conn)
        else:
            # Use the pool
            return DatabaseManager.get_connection()

    def execute(self, query: str, params: tuple = None, fetch: str = None, commit: bool = False) -> Any:
        """
        Simplified execution of a query.
        - fetch: 'one', 'all', or None
        - commit: True for write operations
        """
        with self._get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(query, params)
                if commit:
                    conn.commit()
                if fetch == "one":
                    return cur.fetchone()
                if fetch == "all":
                    return cur.fetchall()
                return None

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
        row = self.execute("SELECT doc_id FROM documents WHERE slug = %s", (doc_identifier,), fetch="one")
        if row:
            return row[0]

        # Fallback: try case-insensitive title match
        row = self.execute("SELECT doc_id FROM documents WHERE LOWER(title) = LOWER(%s)", (doc_identifier,), fetch="one")
        return row[0] if row else None

    def document_exists(self, slug: str) -> bool:
        """Check if a document with given slug exists."""
        row = self.execute("SELECT 1 FROM documents WHERE slug = %s", (slug,), fetch="one")
        return row is not None

    def summaries_exist(self, doc_identifier: int | str) -> bool:
        """Check if summaries exist for a document."""
        doc_id = self._resolve_doc_id(doc_identifier)
        if not doc_id:
            return False

        row = self.execute("SELECT 1 FROM document_summaries WHERE doc_id = %s", (doc_id,), fetch="one")
        return row is not None

    def delete_document(self, doc_identifier: int | str) -> bool:
        """
        Delete a document and all related data.
        All child tables (chapter_summaries, document_summaries, graph_entities,
        graph_relationships, graph_episodes, bm25_index, bm25_doc_lens) use
        ON DELETE CASCADE and are cleaned up automatically.
        """
        doc_id = self._resolve_doc_id(doc_identifier)
        if not doc_id:
            return False

        with self._get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM documents WHERE doc_id = %s", (doc_id,))
                deleted = cur.rowcount > 0
            conn.commit()
        return deleted

    def delete_document_full(self, slug: str) -> tuple[bool, str, int]:
        """
        Delete a document from PostgreSQL and Qdrant.

        Returns:
            (success, message, chunks_deleted)
        """
        result = self.execute(
            "SELECT title, num_chunks FROM documents WHERE slug = %s",
            (slug,), fetch="one",
        )
        if not result:
            return False, f"Document '{slug}' not found", 0

        doc_title = result[0]
        deleted_chunks = result[1] or 0

        # Delete from PostgreSQL (cascade handles all child tables)
        self.delete_document(slug)

        # Delete from Qdrant
        qdrant_success = True
        qdrant_error = ""
        try:
            import os
            from qdrant_client import QdrantClient, models

            qdrant_host = os.getenv("QDRANT_HOST", "localhost")
            qdrant_port = int(os.getenv("QDRANT_PORT", 6333))
            client = QdrantClient(host=qdrant_host, port=qdrant_port)
            client.delete(
                collection_name="book_chunks",
                points_selector=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="id",
                            match=models.MatchText(text=slug),
                        )
                    ]
                ),
            )
        except Exception as e:
            qdrant_success = False
            qdrant_error = str(e)

        if qdrant_success:
            return (
                True,
                f"[SUCCESS] Deleted '{doc_title}' ({deleted_chunks} chunks)",
                deleted_chunks,
            )
        else:
            return (
                True,
                f"[WARNING] Document '{doc_title}' deleted from DB, but Qdrant cleanup failed: {qdrant_error}",
                deleted_chunks,
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
        """Store any document type."""
        clean_metadata = self._sanitize_metadata(metadata)

        with self._get_conn() as conn:
            with conn.cursor() as cur:
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
            conn.commit()
        return doc_id

    def store_summaries(
        self, doc_identifier: int | str, chapter_summaries: list, document_summary: str
    ) -> None:
        """Store chapter and document summaries."""
        doc_id = self._resolve_doc_id(doc_identifier)
        if not doc_id:
            raise ValueError(f"Document not found: {doc_identifier}")

        with self._get_conn() as conn:
            with conn.cursor() as cur:
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
            conn.commit()

    def get_chapter_summary(
        self, doc_identifier: int | str, chapter_id: int
    ) -> str | None:
        """Fetch one chapter summary from DB."""
        doc_id = self._resolve_doc_id(doc_identifier)
        if not doc_id:
            return None

        row = self.execute(
            "SELECT summary FROM chapter_summaries WHERE doc_id = %s AND chapter_id = %s",
            (doc_id, chapter_id),
            fetch="one"
        )
        return row[0] if row else None

    def get_all_chapter_summaries(
        self, doc_identifier: int | str
    ) -> list[tuple[int, str]]:
        """Fetch all chapter summaries for a document."""
        doc_id = self._resolve_doc_id(doc_identifier)
        if not doc_id:
            return []

        return self.execute(
            "SELECT chapter_id, summary FROM chapter_summaries WHERE doc_id = %s ORDER BY chapter_id",
            (doc_id,),
            fetch="all"
        )

    def get_document_summary(self, doc_identifier: int | str) -> str | None:
        """Fetch the overall document summary."""
        doc_id = self._resolve_doc_id(doc_identifier)
        if not doc_id:
            return None

        row = self.execute(
            "SELECT summary FROM document_summaries WHERE doc_id = %s",
            (doc_id,),
            fetch="one"
        )
        return row[0] if row else None

    def store_bm25_index(self, term_freqs: list[tuple[str, str, int, int]], doc_lens: list[tuple[str, int, int]]) -> None:
        """Store BM25 term frequencies and document lengths in bulk.

        Args:
            term_freqs: list of (term, chunk_id, frequency, doc_id)
            doc_lens: list of (chunk_id, doc_len, doc_id)
        """
        with self._get_conn() as conn:
            with conn.cursor() as cur:
                # Store term frequencies
                execute_values(
                    cur,
                    """
                    INSERT INTO bm25_index (term, chunk_id, frequency, doc_id)
                    VALUES %s
                    ON CONFLICT (term, chunk_id) DO UPDATE SET frequency = excluded.frequency
                    """,
                    term_freqs,
                )

                # Store document lengths
                execute_values(
                    cur,
                    """
                    INSERT INTO bm25_doc_lens (chunk_id, doc_len, doc_id)
                    VALUES %s
                    ON CONFLICT (chunk_id) DO UPDATE SET doc_len = excluded.doc_len
                    """,
                    doc_lens,
                )
            conn.commit()

    def get_bm25_stats(self, terms: list[str], doc_slug: str = None) -> dict[str, Any]:
        """Retrieve statistics needed for BM25 scoring."""
        # Resolve slug -> doc_id once for all queries
        doc_id = self._resolve_doc_id(doc_slug) if doc_slug else None

        with self._get_conn() as conn:
            with conn.cursor() as cur:
                # 1. Total document count (N)
                if doc_id:
                    cur.execute("SELECT COUNT(*) FROM bm25_doc_lens WHERE doc_id = %s", (doc_id,))
                else:
                    cur.execute("SELECT COUNT(*) FROM bm25_doc_lens")
                N = cur.fetchone()[0]

                # 2. Document frequencies (df) for the requested terms
                df = {}
                if terms:
                    if doc_id:
                        cur.execute(
                            "SELECT term, COUNT(chunk_id) FROM bm25_index WHERE term = ANY(%s) AND doc_id = %s GROUP BY term",
                            (terms, doc_id),
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
                    if doc_id:
                        cur.execute(
                            """
                            SELECT bi.chunk_id, bi.term, bi.frequency, dl.doc_len
                            FROM bm25_index bi
                            JOIN bm25_doc_lens dl ON bi.chunk_id = dl.chunk_id
                            WHERE bi.term = ANY(%s) AND bi.doc_id = %s
                            """,
                            (terms, doc_id),
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

    # ── Ingest job tracking ──────────────────────────────────────────

    def create_ingest_job(self, job_id: str, slug: str, title: str, doc_type: str) -> None:
        """Record a new ingestion job as 'running'."""
        self.execute(
            """
            INSERT INTO ingest_jobs (job_id, slug, title, doc_type, status)
            VALUES (%s, %s, %s, %s, 'running')
            """,
            (job_id, slug, title, doc_type),
            commit=True,
        )

    def complete_ingest_job(self, job_id: str, result_summary: str) -> None:
        """Mark a job as completed with a one-line summary."""
        self.execute(
            """
            UPDATE ingest_jobs
            SET status = 'completed', result_summary = %s, completed_at = NOW()
            WHERE job_id = %s
            """,
            (result_summary, job_id),
            commit=True,
        )

    def fail_ingest_job(self, job_id: str, error_message: str) -> None:
        """Mark a job as failed with the error message."""
        self.execute(
            """
            UPDATE ingest_jobs
            SET status = 'failed', error_message = %s, completed_at = NOW()
            WHERE job_id = %s
            """,
            (error_message, job_id),
            commit=True,
        )

    def get_latest_ingest_job(self) -> dict | None:
        """Return the most recent ingestion job, or None."""
        row = self.execute(
            """
            SELECT job_id, slug, title, doc_type, status,
                   error_message, result_summary, created_at, completed_at
            FROM ingest_jobs
            ORDER BY created_at DESC
            LIMIT 1
            """,
            fetch="one",
        )
        if not row:
            return None
        return {
            "job_id": row[0],
            "slug": row[1],
            "title": row[2],
            "doc_type": row[3],
            "status": row[4],
            "error_message": row[5],
            "result_summary": row[6],
            "created_at": row[7],
            "completed_at": row[8],
        }

    # Note operations have been moved to src/content/note_store.py (NoteStore class)