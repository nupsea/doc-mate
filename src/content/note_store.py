"""
Note storage operations.

Separated from PgresStore to keep single-responsibility per module.
"""

import json
import time
from src.content.store import PgresStore


class NoteStore(PgresStore):
    """Note-specific database operations. Inherits connection management from PgresStore."""

    def create_note(
        self, title: str, content: str, tags: list[str] = None,
        source_refs: list[dict] = None, is_pinned: bool = False,
    ) -> dict:
        """Create a note as a document + notes row.

        Returns:
            dict with doc_id, slug, note_id
        """
        slug = f"note-{int(time.time()):x}"

        with self._get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO documents (slug, title, author, num_chunks, num_chars, doc_type)
                    VALUES (%s, %s, 'user', 0, %s, 'note')
                    RETURNING doc_id
                    """,
                    (slug, title, len(content)),
                )
                doc_id = cur.fetchone()[0]

                cur.execute(
                    """
                    INSERT INTO notes (doc_id, content, tags, source_refs, is_pinned)
                    VALUES (%s, %s, %s, %s, %s)
                    RETURNING note_id
                    """,
                    (
                        doc_id,
                        content,
                        tags or [],
                        json.dumps(source_refs or []),
                        is_pinned,
                    ),
                )
                note_id = cur.fetchone()[0]
            conn.commit()

        return {"doc_id": doc_id, "slug": slug, "note_id": note_id}

    def update_note(
        self, note_id: int, content: str = None, title: str = None,
        tags: list[str] = None, is_pinned: bool = None,
    ) -> bool:
        """Update note content and/or metadata. Increments version."""
        with self._get_conn() as conn:
            with conn.cursor() as cur:
                sets = []
                params = []
                if content is not None:
                    sets.append("content = %s")
                    params.append(content)
                    sets.append("version = version + 1")
                    sets.append("updated_at = NOW()")
                if tags is not None:
                    sets.append("tags = %s")
                    params.append(tags)
                if is_pinned is not None:
                    sets.append("is_pinned = %s")
                    params.append(is_pinned)

                if sets:
                    params.append(note_id)
                    cur.execute(
                        f"UPDATE notes SET {', '.join(sets)} WHERE note_id = %s",
                        tuple(params),
                    )

                if title is not None:
                    cur.execute(
                        """
                        UPDATE documents SET title = %s
                        WHERE doc_id = (SELECT doc_id FROM notes WHERE note_id = %s)
                        """,
                        (title, note_id),
                    )

                if content is not None:
                    cur.execute(
                        """
                        UPDATE documents SET num_chars = %s
                        WHERE doc_id = (SELECT doc_id FROM notes WHERE note_id = %s)
                        """,
                        (len(content), note_id),
                    )

            conn.commit()
        return True

    def get_note(self, note_id_or_slug) -> dict | None:
        """Fetch a note with full content and document metadata."""
        if isinstance(note_id_or_slug, int):
            where = "n.note_id = %s"
        else:
            where = "d.slug = %s"

        row = self.execute(
            f"""
            SELECT n.note_id, d.doc_id, d.slug, d.title, n.content,
                   n.tags, n.source_refs, n.is_pinned, n.updated_at, n.version,
                   d.num_chunks, d.added_at
            FROM notes n
            JOIN documents d ON n.doc_id = d.doc_id
            WHERE {where}
            """,
            (note_id_or_slug,),
            fetch="one",
        )
        if not row:
            return None

        return {
            "note_id": row[0],
            "doc_id": row[1],
            "slug": row[2],
            "title": row[3],
            "content": row[4],
            "tags": row[5] or [],
            "source_refs": row[6] or [],
            "is_pinned": row[7],
            "updated_at": row[8],
            "version": row[9],
            "num_chunks": row[10],
            "added_at": row[11],
        }

    def list_notes(
        self, tag: str = None, source_slug: str = None,
        search_text: str = None, limit: int = 50,
    ) -> list[dict]:
        """List notes with optional filters."""
        conditions = ["d.doc_type = 'note'"]
        params = []

        if tag:
            conditions.append("n.tags @> ARRAY[%s]::text[]")
            params.append(tag)
        if source_slug:
            conditions.append("n.source_refs @> %s::jsonb")
            params.append(json.dumps([{"slug": source_slug}]))
        if search_text:
            conditions.append("(d.title ILIKE %s OR n.content ILIKE %s)")
            like_pat = f"%{search_text}%"
            params.extend([like_pat, like_pat])

        params.append(limit)
        where = " AND ".join(conditions)

        rows = self.execute(
            f"""
            SELECT n.note_id, d.slug, d.title, n.tags,
                   n.is_pinned, n.updated_at, n.version,
                   LEFT(n.content, 120) as preview
            FROM notes n
            JOIN documents d ON n.doc_id = d.doc_id
            WHERE {where}
            ORDER BY n.is_pinned DESC, n.updated_at DESC
            LIMIT %s
            """,
            tuple(params),
            fetch="all",
        )

        return [
            {
                "note_id": r[0],
                "slug": r[1],
                "title": r[2],
                "tags": r[3] or [],
                "is_pinned": r[4],
                "updated_at": r[5],
                "version": r[6],
                "preview": r[7],
            }
            for r in (rows or [])
        ]

    def delete_note(self, note_id: int) -> bool:
        """Delete a note. CASCADE handles all cleanup."""
        with self._get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT doc_id FROM notes WHERE note_id = %s", (note_id,))
                row = cur.fetchone()
                if not row:
                    return False

                cur.execute("DELETE FROM documents WHERE doc_id = %s", (row[0],))
                deleted = cur.rowcount > 0
            conn.commit()
        return deleted

    def get_notes_referencing_doc(self, slug: str) -> list[dict]:
        """Find all notes that cite a given document slug."""
        rows = self.execute(
            """
            SELECT n.note_id, d.slug, d.title, n.tags, n.updated_at
            FROM notes n
            JOIN documents d ON n.doc_id = d.doc_id
            WHERE n.source_refs @> %s::jsonb
            ORDER BY n.updated_at DESC
            """,
            (json.dumps([{"slug": slug}]),),
            fetch="all",
        )
        return [
            {
                "note_id": r[0],
                "slug": r[1],
                "title": r[2],
                "tags": r[3] or [],
                "updated_at": r[4],
            }
            for r in (rows or [])
        ]
