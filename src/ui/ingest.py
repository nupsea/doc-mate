"""
Document ingestion interface component - supports books, scripts, conversations, tech docs, reports.

Backward compatible with document ingestion.
"""

import asyncio
import sys
import threading
import uuid
import gradio as gr
from pathlib import Path
from src.content.store import PgresStore
from src.flows.document_ingest import ingest_document
from src.ui.utils import (
    validate_slug,
    extract_chapter_info_from_chunks,
    format_document_list,
    get_available_documents,
    delete_document,
)
from src.ui.pattern_builder import build_pattern_from_example, validate_pattern_on_file
from src.flows.ingest_profiles import PROFILES, DEFAULT_PROFILE, check_ollama_available


class _StreamCapture:
    """Thread-safe capture of stdout writes for streaming to Gradio UI."""

    def __init__(self, original_stdout):
        self._original = original_stdout
        self._buffer: list[str] = []
        self._lock = threading.Lock()

    def write(self, text):
        self._original.write(text)
        if text.strip():
            with self._lock:
                self._buffer.append(text)

    def flush(self):
        self._original.flush()

    def drain(self) -> str:
        """Return and clear buffered lines."""
        with self._lock:
            lines = list(self._buffer)
            self._buffer.clear()
        return "".join(lines)


def test_chapter_pattern(file, chapter_example: str):
    """Test pattern on uploaded file before ingestion."""
    if not file:
        return "Please upload a file first"

    if not chapter_example.strip():
        return "Please provide a chapter example (e.g., 'CHAPTER 2' or 'BOOK II')"

    try:
        file_path = Path(file.name)

        # Build pattern from example
        pattern, desc = build_pattern_from_example(chapter_example)

        if not pattern:
            return f"Could not build pattern: {desc}"

        # Validate against file
        success, message, matches = validate_pattern_on_file(pattern, str(file_path))

        output = f"Example: '{chapter_example}'\n"
        output += f"Generated pattern: {pattern}\n"
        output += f"Description: {desc}\n\n"

        if success:
            output += f"[SUCCESS] {message}\n\n"
            output += "Sample matches:\n"
            for i, (line_num, text) in enumerate(matches[:5], 1):
                output += f"  {i}. Line {line_num}: {text[:60]}\n"
            output += "\nPattern looks good! You can proceed with ingestion."
        else:
            output += f"[FAILED] {message}\n\n"
            output += "Please try a different example or check your file format."

        return output

    except Exception as e:
        return f"Error testing pattern: {str(e)}"


async def ingest_new_document(
    file,
    title: str,
    author: str,
    slug: str,
    skip_chapters: bool,
    chapter_example: str,
    force_update: bool,
    doc_type: str = 'book',
    ephemeral: bool = False,
    profile_name: str = None,
):
    """Handle document ingestion from UI (all types)."""
    if not file:
        return {
            "output": "Error: Please upload a file",
            "status": "[ERROR] Error",
            "clear_inputs": False,
        }

    if not title.strip():
        return {
            "output": "Error: Please provide a document title",
            "status": "[ERROR] Error",
            "clear_inputs": False,
        }

    slug = slug.strip().lower()

    if not slug:
        return {
            "output": "Error: Please provide a slug",
            "status": "[ERROR] Error",
            "clear_inputs": False,
        }

    # Only validate chapter example for books
    if doc_type == 'book' and not skip_chapters and not chapter_example.strip():
        return {
            "output": "Error: Please provide a chapter example or enable 'Skip chapter detection'",
            "status": "[ERROR] Error",
            "clear_inputs": False,
        }

    # Validate slug (skip duplicate check if force_update is enabled)
    if not force_update:
        is_valid, error_msg = validate_slug(slug)
        if not is_valid:
            return {
                "output": f"Error: {error_msg}",
                "status": "[ERROR] Error",
                "clear_inputs": False,
            }

    # Create a persistent job record so the UI can show status after reconnect
    job_id = uuid.uuid4().hex[:12]
    job_store = PgresStore()

    try:
        file_path = Path(file.name)

        # Handle pattern building (only for books)
        if doc_type == 'book':
            if skip_chapters:
                pattern = None
                output = "[SKIP] Chapter detection disabled - using automatic chunking\n\n"
            else:
                # Build pattern from example
                pattern, desc = build_pattern_from_example(chapter_example)
                output = f"Building pattern from example: '{chapter_example}'\n"
                output += f"Generated pattern: {pattern}\n"
                output += f"Description: {desc}\n\n"

                if not pattern:
                    return {
                        "output": output + f"Error: {desc}",
                        "status": "[ERROR] Pattern Error",
                        "clear_inputs": False,
                    }

                # Validate pattern
                success, message, matches = validate_pattern_on_file(
                    pattern, str(file_path)
                )
                output += f"Pattern validation: {message}\n\n"

                if not success:
                    return {
                        "output": output
                        + "Pattern validation failed. Please try a different example.",
                        "status": "[ERROR] Validation Failed",
                        "clear_inputs": False,
                    }
        else:
            # For non-book types, no pattern needed (parsers handle structure internally)
            pattern = None
            output = f"[INFO] Document type: {doc_type} - using built-in parser\n\n"

        output += "[RUNNING] Starting ingestion...\n"

        if ephemeral:
            output += "[EPHEMERAL] Ephemeral mode enabled - no traces will be created\n"

        # Record job as running *before* the actual work begins
        job_store.create_ingest_job(job_id, slug, title.strip(), doc_type)

        # Run ingestion (use ingest_document for all types)
        result = await ingest_document(
            slug=slug,
            file_path=str(file_path),
            title=title,
            doc_type=doc_type,
            author=author or None,
            split_pattern=pattern,
            force_update=force_update,
            ephemeral=ephemeral,
            profile_name=profile_name,
        )

        output += f"\n[SUCCESS] {doc_type.title()} ingested:\n"
        output += f"- Slug: {result['slug']}\n"
        output += f"- Title: {result['title']}\n"
        output += f"- Sections: {result['chapters']}\n"
        output += f"- Chunks: {result['chunks']}\n"
        output += f"- Search indexed: {result['search_indexed']}\n\n"

        # Analyze chunks to verify structure detection
        output += "Analyzing indexed chunks...\n"
        chunk_info = extract_chapter_info_from_chunks(slug)

        structure_detail = ""
        if chunk_info["status"] == "success":
            output += f"- Total chunks indexed: {chunk_info['total_chunks']}\n"
            output += f"- First chunk ID: {chunk_info['first_chunk']}\n"
            output += f"- Last chunk ID: {chunk_info['last_chunk']}\n\n"

            # Conversations use sequential chunk IDs (no section grouping in IDs),
            # so section verification is only meaningful for books/docs.
            if doc_type == "conversation":
                output += f"[OK] Indexed {chunk_info['total_chunks']} chunks, {result['chapters']} arc summaries. Ingestion successful."
                structure_detail = f"{result['chapters']} arcs, {chunk_info['total_chunks']} chunks"
            elif chunk_info["total_sections"] == result["chapters"]:
                output += f"- Total sections detected: {chunk_info['total_sections']}\n"
                output += f"- Section range: {chunk_info['section_range']}\n"
                output += "[OK] Section count matches! Ingestion successful."
                structure_detail = f"Sections: {', '.join(chunk_info['sections'])}"
            else:
                output += f"- Total sections detected: {chunk_info['total_sections']}\n"
                output += f"- Section range: {chunk_info['section_range']}\n"
                output += "[WARNING] Section count mismatch!\n"
                output += f"Expected: {result['chapters']}, Found in index: {chunk_info['total_sections']}"
                structure_detail = f"Mismatch: {chunk_info['total_sections']} sections"
        else:
            output += f"[ERROR] Error analyzing chunks: {chunk_info['message']}"
            structure_detail = "Analysis failed"

        # Mark job completed with a short summary
        summary = f"chunks={result['chunks']}, sections={result['chapters']}, indexed={result['search_indexed']}"
        job_store.complete_ingest_job(job_id, summary)

        return {
            "output": output,
            "status": f"[COMPLETE] Ingestion Complete ({result['chapters']} sections, {result['chunks']} chunks)",
            "chapter_detail": structure_detail,
            "clear_inputs": True,
        }

    except Exception as e:
        import traceback
        traceback.print_exc()  # This will print full traceback to docker logs

        # Persist the failure so the banner can show it after reconnect
        try:
            job_store.fail_ingest_job(job_id, str(e))
        except Exception:
            pass  # Best effort -- don't mask the original error

        return {
            "output": f"[ERROR] Error during ingestion: {str(e)}",
            "status": "[ERROR] Ingestion Failed",
            "clear_inputs": False,
        }


def _format_job_banner(job: dict | None) -> str:
    """Return a Markdown string describing the latest ingest job, or empty."""
    if not job:
        return ""
    from datetime import datetime

    created = job["created_at"]
    now = datetime.now(created.tzinfo) if created.tzinfo else datetime.now()
    mins_ago = max(int((now - created).total_seconds() / 60), 0)
    time_label = f"{mins_ago} min ago" if mins_ago < 120 else f"{mins_ago // 60}h ago"

    slug = job["slug"]
    title = job["title"] or slug

    if job["status"] == "running":
        return f"**Ingestion in progress:** '{title}' ({slug}) is still running... (started {time_label})"
    if job["status"] == "completed":
        summary = job["result_summary"] or ""
        return f"**Last ingestion:** '{title}' ({slug}) completed -- {summary} ({time_label})"
    if job["status"] == "failed":
        err = job["error_message"] or "unknown error"
        return f"**Last ingestion failed:** '{title}' ({slug}) -- {err} ({time_label})"
    return ""


def create_ingest_interface():
    """Create the document ingestion tab interface."""
    from datetime import datetime

    with gr.Column():
        gr.Markdown("### Upload and Index a New Document")

        last_job_banner = gr.Markdown(value="", elem_id="ingest-job-banner")

        with gr.Row():
            with gr.Column(scale=2):
                # Document type selector
                doc_type_selector = gr.Dropdown(
                    choices=["book", "script", "conversation", "tech_doc", "report"],
                    value="book",
                    label="Document Type",
                    info="Select the type of document you're uploading"
                )

                # Ingestion profile selector
                profile_choices = [(p.label, key) for key, p in PROFILES.items()]
                profile_selector = gr.Radio(
                    choices=profile_choices,
                    value=DEFAULT_PROFILE,
                    label="Ingestion Profile",
                    info="Controls speed / quality / cost tradeoffs",
                )

                profile_description = gr.Markdown(
                    value=PROFILES[DEFAULT_PROFILE].description,
                )

                time_estimate_display = gr.Textbox(
                    label="Estimated Time",
                    value="Upload a file to see time estimate",
                    lines=1,
                    interactive=False,
                )

                file_upload = gr.File(
                    label="Upload Document File (.txt or .pdf)", file_types=[".txt", ".pdf"]
                )

                title_input = gr.Textbox(
                    label="Document Title", placeholder="The Meditations", info="Required"
                )

                author_input = gr.Textbox(
                    label="Author", placeholder="Marcus Aurelius", info="Optional"
                )

                slug_input = gr.Textbox(
                    label="Document Slug (unique identifier)",
                    placeholder="mma",
                    info="2-20 chars, lowercase, letters/numbers/-/_ only",
                    max_lines=1,
                )

                skip_chapters_check = gr.Checkbox(
                    label="Skip section detection (use auto-chunking)",
                    value=False,
                    info="Enable if document has no clear chapters or complex structure",
                )

                chapter_example_input = gr.Textbox(
                    label="Section/Chapter Pattern Example",
                    placeholder="e.g., CHAPTER I. or II. or BOOK II",
                    info="Enter any section heading from your document, then test pattern. Examples: 'CHAPTER I.', 'BOOK II', 'II.'",
                    lines=1,
                    visible=True,
                )

                test_pattern_btn = gr.Button("Test Pattern", size="sm", visible=True)

                pattern_test_output = gr.Textbox(
                    label="Pattern Test Results",
                    lines=6,
                    interactive=False,
                    visible=True,
                )

                nested_structure_note = gr.Markdown(
                    """
                **Note:** For nested structures (PART > CHAPTER), use the higher level pattern (e.g., `PART I.` instead of `CHAPTER I.`)
                """,
                    visible=True,
                )

                force_update_check = gr.Checkbox(
                    label="Force update if slug exists",
                    value=False,
                    info="Overwrite existing document",
                )

                ephemeral_mode_check = gr.Checkbox(
                    label="Ephemeral mode (no traces)",
                    value=False,
                    info="Disable tracing during summarization - no Phoenix traces will be created",
                )

                ingest_btn = gr.Button("Ingest Document", variant="primary", size="lg")

            with gr.Column(scale=1):
                gr.Markdown("### Current Library")

                library_timestamp = gr.Textbox(
                    value=f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                    lines=1,
                    interactive=False,
                    show_label=False,
                )

                doc_list_display = gr.Dataframe(
                    headers=["Slug", "Title", "Author", "Chunks", "Added"],
                    datatype=["str", "str", "str", "number", "str"],
                    interactive=False,
                    wrap=True,
                    column_widths=["15%", "25%", "20%", "12%", "28%"],
                    max_height=800,
                )

                gr.Markdown("#### Delete Document")

                delete_slug_input = gr.Textbox(
                    label="Document Slug to Delete",
                    placeholder="Enter slug (e.g., mma)",
                    lines=1,
                )

                delete_output = gr.Textbox(
                    label="Delete Status", lines=4, interactive=False
                )

                delete_btn = gr.Button(
                    "Confirm Delete", variant="stop", size="sm", visible=False
                )

        # Status indicator
        status_display = gr.Textbox(
            label="Status", value="Ready", lines=1, interactive=False
        )

        ingest_output = gr.Textbox(label="Ingestion Log", lines=12, interactive=False)

        structure_info = gr.Textbox(
            label="Structure Verification", lines=2, interactive=False
        )

        # Event handlers

        # Toggle chapter pattern fields visibility based on doc_type and skip_chapters
        def toggle_chapter_fields(doc_type, skip_chapters):
            # Only show chapter fields for books (and only if skip_chapters is False)
            is_book = doc_type == 'book'
            visible = is_book and not skip_chapters

            return (
                gr.update(visible=visible),  # chapter_example_input
                gr.update(visible=visible),  # test_pattern_btn
                gr.update(visible=visible),  # pattern_test_output
                gr.update(visible=visible),  # nested_structure_note
                gr.update(visible=is_book),  # skip_chapters_check (only for books)
            )

        # Update visibility when document type changes
        doc_type_selector.change(
            toggle_chapter_fields,
            [doc_type_selector, skip_chapters_check],
            [
                chapter_example_input,
                test_pattern_btn,
                pattern_test_output,
                nested_structure_note,
                skip_chapters_check,
            ],
        )

        # Update visibility when skip_chapters checkbox changes
        skip_chapters_check.change(
            toggle_chapter_fields,
            [doc_type_selector, skip_chapters_check],
            [
                chapter_example_input,
                test_pattern_btn,
                pattern_test_output,
                nested_structure_note,
                skip_chapters_check,
            ],
        )

        # Update profile description and time estimate
        def update_profile_info(profile_name, file, doc_type):
            profile = PROFILES.get(profile_name, PROFILES[DEFAULT_PROFILE])
            desc = profile.description

            needs_local = profile.summary_provider == "local" or profile.graph_provider == "local"
            if needs_local:
                try:
                    if not check_ollama_available():
                        desc += "\n\n**Warning:** Ollama is not reachable. Start Ollama or choose an OpenAI profile."
                except Exception:
                    desc += "\n\n**Warning:** Could not check Ollama availability."

            time_str = "Upload a file to see time estimate"
            if file:
                try:
                    file_size = Path(file.name).stat().st_size
                    est_chunks = max(1, file_size // 4 // 500)
                    min_s, max_s = profile.estimate_seconds(est_chunks, doc_type or "book")
                    min_m, max_m = max(1, min_s // 60), max(1, max_s // 60)
                    time_str = f"~{min_m}-{max_m} min (~{est_chunks} chunks)"
                except Exception:
                    time_str = "Could not estimate"

            return desc, time_str

        profile_selector.change(
            update_profile_info,
            [profile_selector, file_upload, doc_type_selector],
            [profile_description, time_estimate_display],
        )

        file_upload.change(
            update_profile_info,
            [profile_selector, file_upload, doc_type_selector],
            [profile_description, time_estimate_display],
        )

        test_pattern_btn.click(
            test_chapter_pattern,
            [file_upload, chapter_example_input],
            pattern_test_output,
        )

        async def handle_ingest(
            doc_type, profile_name, file, title, author, slug, skip_chap, chapter_ex, force, ephemeral
        ):
            """Streaming ingestion handler -- yields progress every 2s to keep UI alive."""
            # Helper to build a yield-tuple that only updates log + status
            keep = gr.update()
            def _progress(log_text, status_text):
                return (
                    log_text, status_text, keep,   # log, status, structure
                    keep, keep, keep, keep, keep, keep,  # inputs unchanged
                    keep, keep,                          # library unchanged
                    keep,                                # banner unchanged
                )

            # Start capturing stdout so we can relay pipeline prints to the UI
            capture = _StreamCapture(sys.stdout)
            old_stdout = sys.stdout
            sys.stdout = capture

            # Launch ingestion as a concurrent task
            task = asyncio.ensure_future(
                ingest_new_document(
                    file, title, author, slug, skip_chap, chapter_ex, force, doc_type, ephemeral,
                    profile_name=profile_name,
                )
            )

            log_so_far = ""
            try:
                # Poll for progress while ingestion runs
                while not task.done():
                    await asyncio.sleep(2)
                    new_lines = capture.drain()
                    if new_lines:
                        log_so_far += new_lines + "\n"
                        yield _progress(log_so_far, "[RUNNING] Ingestion in progress...")

                # Drain any final output
                new_lines = capture.drain()
                if new_lines:
                    log_so_far += new_lines + "\n"
            finally:
                sys.stdout = old_stdout

            # Get the result (may raise if ingestion failed)
            result = task.result()

            # Refresh library list
            new_list = format_document_list(get_available_documents())
            new_timestamp = (
                f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            )

            final_log = log_so_far + "\n" + result["output"]

            # Refresh banner after ingestion
            try:
                banner = _format_job_banner(PgresStore().get_latest_ingest_job())
            except Exception:
                banner = ""

            if result["clear_inputs"]:
                yield (
                    final_log,
                    result["status"],
                    result.get("chapter_detail", ""),
                    None,  # Clear file
                    "",  # Clear title
                    "",  # Clear author
                    "",  # Clear slug
                    "",  # Clear chapter example
                    "",  # Clear pattern test
                    new_list,
                    new_timestamp,
                    banner,
                )
            else:
                yield (
                    final_log,
                    result["status"],
                    result.get("chapter_detail", ""),
                    keep, keep, keep, keep, keep, keep,
                    new_list,
                    new_timestamp,
                    banner,
                )

        ingest_btn.click(
            handle_ingest,
            [
                doc_type_selector,
                profile_selector,
                file_upload,
                title_input,
                author_input,
                slug_input,
                skip_chapters_check,
                chapter_example_input,
                force_update_check,
                ephemeral_mode_check,
            ],
            [
                ingest_output,
                status_display,
                structure_info,
                file_upload,
                title_input,
                author_input,
                slug_input,
                chapter_example_input,
                pattern_test_output,
                doc_list_display,
                library_timestamp,
                last_job_banner,
            ],
        )

        # Delete document handler with confirmation state
        delete_pending_slug = gr.State(None)

        def request_delete_confirmation(slug):
            """First step: show confirmation message"""
            slug = slug.strip().lower()

            if not slug:
                return (
                    "[ERROR] Please enter a document slug",
                    None,  # No pending slug
                    gr.update(visible=False),  # Hide confirm button
                )

            # Get document info
            docs = get_available_documents()
            doc_info = next((d for d in docs if d[0] == slug), None)

            if not doc_info:
                return (
                    f"[ERROR] Document '{slug}' not found",
                    None,
                    gr.update(visible=False),
                )

            doc_slug, doc_title, doc_author, num_chunks, _ = doc_info
            author_str = f" by {doc_author}" if doc_author else ""

            confirm_msg = (
                f"[CONFIRM?] Delete '{doc_title}'{author_str}? ({num_chunks} chunks)\n"
            )
            confirm_msg += "This action cannot be undone.\n\n"
            confirm_msg += "Click 'Confirm Delete' button below to proceed."

            return (
                confirm_msg,
                slug,  # Store slug for confirmation
                gr.update(visible=True),  # Show confirm button
            )

        def confirm_delete(pending_slug):
            """Second step: actually delete after confirmation"""
            if not pending_slug:
                return (
                    "[ERROR] No deletion pending",
                    gr.update(),
                    gr.update(),
                    "",
                    None,
                    gr.update(visible=False),
                )

            output = f"Deleting document '{pending_slug}'...\n\n"
            success, message, chunks_deleted = delete_document(pending_slug)

            # Always refresh document list after deletion attempt
            new_list = format_document_list(get_available_documents())
            new_timestamp = (
                f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            )

            return (
                output + message,
                new_list,
                new_timestamp,
                "",  # Clear slug input
                None,  # Clear pending slug
                gr.update(visible=False),  # Hide confirm button
            )

        delete_slug_input.change(
            request_delete_confirmation,
            [delete_slug_input],
            [delete_output, delete_pending_slug, delete_btn],
        )

        delete_btn.click(
            confirm_delete,
            [delete_pending_slug],
            [
                delete_output,
                doc_list_display,
                library_timestamp,
                delete_slug_input,
                delete_pending_slug,
                delete_btn,
            ],
        )

    return doc_list_display, last_job_banner