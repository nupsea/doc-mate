"""
Notes tab UI -- browse, create, edit, and search notes.
"""

import asyncio
import json

import gradio as gr
from src.content.note_store import NoteStore


def _load_note_list(tag_filter=None, search_text=None):
    """Load notes for the list panel."""
    store = NoteStore()
    tag = tag_filter if tag_filter and tag_filter != "All" else None
    search = search_text if search_text and search_text.strip() else None
    notes = store.list_notes(tag=tag, search_text=search)

    rows = []
    for n in notes:
        tags_str = ", ".join(n["tags"]) if n["tags"] else ""
        updated = n["updated_at"].strftime("%Y-%m-%d %H:%M") if n["updated_at"] else ""
        pin = "[*] " if n["is_pinned"] else ""
        rows.append([n["note_id"], f"{pin}{n['title']}", tags_str, updated])
    return rows


def _get_all_tags():
    """Collect all unique tags across notes."""
    store = NoteStore()
    rows = store.execute(
        "SELECT DISTINCT unnest(tags) AS tag FROM notes ORDER BY tag",
        fetch="all",
    )
    tags = [r[0] for r in rows] if rows else []
    return ["All"] + tags


def _format_sources_md(source_refs):
    """Format source references as markdown."""
    if not source_refs:
        return "*No source references*"
    refs = source_refs if isinstance(source_refs, list) else json.loads(source_refs)
    if not refs:
        return "*No source references*"
    lines = ["**Source passages:**"]
    for ref in refs[:15]:
        slug = ref.get("slug", "?")
        chunk_id = ref.get("chunk_id", "?")
        snippet = ref.get("snippet", "")[:80]
        query = ref.get("query", "")
        line = f"- `{slug}` / `{chunk_id}`"
        if snippet:
            line += f": {snippet}..."
        if query:
            line += f" (query: *{query[:50]}*)"
        lines.append(line)
    return "\n".join(lines)


def create_notes_interface():
    """Create the Notes tab interface."""

    with gr.Row():
        # --- Left panel: note list ---
        with gr.Column(scale=1, min_width=320):
            gr.Markdown("### Your Notes")

            with gr.Row():
                new_note_btn = gr.Button("+ New Note", variant="primary", size="sm")

            with gr.Row():
                search_box = gr.Textbox(
                    placeholder="Search notes...",
                    show_label=False,
                    scale=3,
                )
                tag_filter = gr.Dropdown(
                    choices=["All"],
                    value="All",
                    label="Tag",
                    scale=1,
                    min_width=100,
                )

            note_table = gr.Dataframe(
                headers=["ID", "Title", "Tags", "Updated"],
                datatype=["number", "str", "str", "str"],
                interactive=False,
                max_height=600,
                column_widths=["10%", "45%", "25%", "20%"],
            )

        # --- Right panel: note editor ---
        with gr.Column(scale=2):
            gr.Markdown("### Note Editor")

            # Hidden state for tracking the currently loaded note
            current_note_id = gr.State(value=None)

            note_title_input = gr.Textbox(label="Title", placeholder="Note title...")
            note_tags_input = gr.Textbox(
                label="Tags (comma-separated)",
                placeholder="e.g. odyssey, themes, characters",
            )
            note_content_input = gr.Textbox(
                label="Content",
                lines=15,
                placeholder="Write your note in markdown...",
            )

            note_sources_md = gr.Markdown(value="*No source references*")

            with gr.Row():
                save_btn = gr.Button("Save", variant="primary")
                reindex_btn = gr.Button("Re-index")
                delete_btn = gr.Button("Delete", variant="stop")
                query_btn = gr.Button("Query Sources")

            editor_status = gr.Textbox(
                label="Status", interactive=False, visible=True, value=""
            )

    # ── Event handlers ──────────────────────────────────────────────

    def refresh_list(tag_val=None, search_val=None):
        """Refresh note list and tag filter choices."""
        rows = _load_note_list(tag_filter=tag_val, search_text=search_val)
        tags = _get_all_tags()
        return rows, gr.update(choices=tags, value=tag_val or "All")

    def on_filter_change(tag_val, search_val):
        rows = _load_note_list(tag_filter=tag_val, search_text=search_val)
        return rows

    def load_note(evt: gr.SelectData, table_data):
        """Load a note into the editor when clicked in the table."""
        import pandas as pd
        if isinstance(table_data, pd.DataFrame):
            if table_data.empty or evt.index[0] >= len(table_data):
                return gr.update(), gr.update(), gr.update(), gr.update(), None, ""
            row = table_data.iloc[evt.index[0]]
        else:
            if not table_data or evt.index[0] >= len(table_data):
                return gr.update(), gr.update(), gr.update(), gr.update(), None, ""
            row = table_data[evt.index[0]]
        note_id = int(row[0])

        store = NoteStore()
        note = store.get_note(note_id)
        if not note:
            return gr.update(), gr.update(), gr.update(), gr.update(), None, "Note not found"

        tags_str = ", ".join(note["tags"]) if note["tags"] else ""
        sources_md = _format_sources_md(note.get("source_refs"))

        return (
            gr.update(value=note["title"]),
            gr.update(value=tags_str),
            gr.update(value=note["content"]),
            gr.update(value=sources_md),
            note_id,
            f"Loaded: {note['title']} (v{note['version']})",
        )

    def new_note():
        """Clear editor for a new note."""
        return (
            gr.update(value=""),
            gr.update(value=""),
            gr.update(value=""),
            gr.update(value="*No source references*"),
            None,
            "New note -- type content and click Save",
        )

    async def save_note(note_id, title, content, tags_str):
        """Save or create a note."""
        if not title or not title.strip():
            return note_id, "Title is required.", gr.update(), gr.update()
        if not content or not content.strip():
            return note_id, "Content is required.", gr.update(), gr.update()

        tags = [t.strip() for t in tags_str.split(",") if t.strip()] if tags_str else []
        store = NoteStore()

        if note_id is not None:
            # Update existing
            store.update_note(note_id, content=content, title=title, tags=tags)
            slug = store.get_note(note_id)["slug"]
            status = f"Updated: {title}"
        else:
            # Create new
            result = store.create_note(title, content, tags=tags)
            note_id = result["note_id"]
            slug = result["slug"]
            status = f"Created: {title} ({slug})"

        # Index in background
        from src.flows.note_ingest import index_note
        asyncio.create_task(index_note(slug, content))

        # Refresh list
        rows = _load_note_list()
        tag_choices = _get_all_tags()
        return note_id, status, rows, gr.update(choices=tag_choices)

    def delete_note(note_id):
        """Delete a note with confirmation via double-click pattern."""
        if note_id is None:
            return note_id, "No note selected.", gr.update(), gr.update()

        store = NoteStore()
        note = store.get_note(note_id)
        if not note:
            return None, "Note not found.", gr.update(), gr.update()

        title = note["title"]
        store.delete_note(note_id)

        # Also clean Qdrant vectors
        try:
            from src.flows.document_ingest import _cleanup_qdrant_vectors
            _cleanup_qdrant_vectors(note["slug"])
        except Exception:
            pass

        rows = _load_note_list()
        tag_choices = _get_all_tags()
        return (
            None,
            f"Deleted: {title}",
            rows,
            gr.update(choices=tag_choices),
        )

    async def reindex_note(note_id):
        """Re-index a note's search indexes."""
        if note_id is None:
            return "No note selected."

        store = NoteStore()
        note = store.get_note(note_id)
        if not note:
            return "Note not found."

        from src.flows.note_ingest import index_note
        await index_note(note["slug"], note["content"])
        return f"Re-indexed: {note['title']} ({note['slug']})"

    def query_sources(content):
        """Placeholder: in future, search for related passages across all documents."""
        if not content or not content.strip():
            return "No content to query."
        return "Query Sources will search for related passages across your documents. (Coming in Phase 2)"

    # ── Wire events ─────────────────────────────────────────────────

    # Note list interactions
    note_table.select(
        load_note,
        [note_table],
        [note_title_input, note_tags_input, note_content_input,
         note_sources_md, current_note_id, editor_status],
    )

    # Filtering
    tag_filter.change(on_filter_change, [tag_filter, search_box], [note_table])
    search_box.change(on_filter_change, [tag_filter, search_box], [note_table])

    # New note
    new_note_btn.click(
        new_note,
        None,
        [note_title_input, note_tags_input, note_content_input,
         note_sources_md, current_note_id, editor_status],
    )

    # Save
    save_btn.click(
        save_note,
        [current_note_id, note_title_input, note_content_input, note_tags_input],
        [current_note_id, editor_status, note_table, tag_filter],
    )

    # Delete
    delete_btn.click(
        delete_note,
        [current_note_id],
        [current_note_id, editor_status, note_table, tag_filter],
    )

    # Re-index
    reindex_btn.click(reindex_note, [current_note_id], [editor_status])

    # Query sources
    query_btn.click(query_sources, [note_content_input], [editor_status])

    # Load initial data
    def load_initial():
        rows = _load_note_list()
        tags = _get_all_tags()
        return rows, gr.update(choices=tags)

    return note_table, tag_filter, load_initial
