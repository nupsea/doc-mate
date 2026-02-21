"""
Chat interface component.
"""

import os

import gradio as gr
from src.monitoring.metrics import metrics_collector


# Store query_id and source_refs for each chat turn (message index -> {query_id, source_refs})
query_id_map = {}
source_refs_map = {}


async def respond(message, chat_history, selected_doc, selected_provider, selected_model, privacy_mode, ui):
    """Handle chat interactions.

    Yields:
        (chat_history, message_text, feedback_row_update,
         save_note_accordion_update, note_title, note_content, note_sources_md)
    """
    if not message.strip():
        yield (chat_history, message, gr.update(visible=False),
               gr.update(visible=False), gr.update(), gr.update(), gr.update())
        return

    # Update provider/model if changed (with proper cleanup)
    settings_changed, was_ephemeral, is_ephemeral = await ui.set_provider_and_model(selected_provider, selected_model, privacy_mode)

    # Clear conversation history ONLY when switching FROM ephemeral TO non-ephemeral
    if settings_changed and was_ephemeral and not is_ephemeral:
        print("[UI] Switching from ephemeral to non-ephemeral mode - clearing conversation history to preserve privacy")
        chat_history = []
        # Clear query_id and source_refs maps as well
        query_id_map.clear()
        source_refs_map.clear()
    elif settings_changed:
        print("[UI] Settings changed but preserving conversation history for continuity")

    # Add user message with loading indicator for bot
    chat_history.append([message, "Thinking..."])

    # Keep message in textbox during processing
    yield (chat_history, message, gr.update(visible=False),
           gr.update(visible=False), gr.update(), gr.update(), gr.update())

    # Get bot response
    bot_response, query_id, source_refs = await ui.chat(message, chat_history[:-1], selected_doc)

    # Update with actual response
    chat_history[-1][1] = bot_response

    # Store query_id and source_refs for this interaction
    msg_idx = len(chat_history) - 1
    if query_id:
        query_id_map[msg_idx] = query_id
    if source_refs:
        source_refs_map[msg_idx] = source_refs

    # Pre-fill save-as-note fields
    default_title = message[:60].strip()
    sources_md = _format_source_refs_md(source_refs)

    # Clear textbox, show feedback + save-note accordion
    yield (chat_history, "", gr.update(visible=True, value=None),
           gr.update(visible=True, open=False),
           gr.update(value=default_title),
           gr.update(value=bot_response),
           gr.update(value=sources_md))


def _format_source_refs_md(source_refs: list) -> str:
    """Format source references as markdown for display."""
    if not source_refs:
        return "*No source references*"
    lines = ["**Source passages:**"]
    for ref in source_refs[:10]:  # Cap at 10 for display
        slug = ref.get("slug", "?")
        chunk_id = ref.get("chunk_id", "?")
        snippet = ref.get("snippet", "")[:80]
        meta_parts = []
        if ref.get("timestamp"):
            meta_parts.append(ref["timestamp"])
        if ref.get("speakers"):
            meta_parts.append(", ".join(ref["speakers"]))
        meta_str = f" ({' | '.join(meta_parts)})" if meta_parts else ""
        lines.append(f"- `{slug}`{meta_str} / `{chunk_id}`: {snippet}...")
    if len(source_refs) > 10:
        lines.append(f"- *...and {len(source_refs) - 10} more*")
    return "\n".join(lines)


async def save_as_note(title, content, tags_str, chat_history):
    """Save the current response as a note."""
    if not title or not content:
        return gr.update(visible=True, value="Title and content are required.")

    from src.content.note_store import NoteStore
    from src.flows.note_ingest import index_note

    # Parse tags
    tags = [t.strip() for t in tags_str.split(",") if t.strip()] if tags_str else []

    # Get source refs from the last response
    last_idx = len(chat_history) - 1 if chat_history else -1
    refs = source_refs_map.get(last_idx, [])

    # Also capture the query that generated this response
    query = chat_history[last_idx][0] if last_idx >= 0 and chat_history else ""
    enriched_refs = []
    for ref in refs:
        enriched_ref = dict(ref)
        enriched_ref["query"] = query
        enriched_refs.append(enriched_ref)

    try:
        store = NoteStore()
        result = store.create_note(title, content, tags=tags, source_refs=enriched_refs)
        slug = result["slug"]

        # Index in background (non-blocking)
        import asyncio
        asyncio.create_task(index_note(slug, content))

        return gr.update(visible=True, value=f"Note saved: '{title}' ({slug})")
    except Exception as e:
        print(f"[SAVE NOTE] Error: {e}")
        return gr.update(visible=True, value=f"Error saving note: {e}")


def submit_feedback(rating, chat_history):
    """Submit user feedback for the last bot response."""
    if not chat_history or rating is None:
        return gr.update(visible=False)

    # Get query_id for last message
    last_idx = len(chat_history) - 1
    query_id = query_id_map.get(last_idx)

    if query_id:
        metrics_collector.update_user_feedback(query_id, rating)
        return gr.update(visible=False, value="Thanks for your feedback!")

    return gr.update(visible=False)


def update_model_choices(provider, privacy_mode):
    """Update model dropdown based on selected provider and privacy mode."""
    # Internal or Private modes force local LLM
    force_local = privacy_mode in ["internal", "private"]

    if force_local or provider == "local":
        return gr.update(
            choices=[
                ("Granite 3.2 8B (Recommended)", "granite3.2:8b"),
                ("Llama 3.2 3B (Faster, less accurate)", "llama3.2:3b"),
            ],
            value="granite3.2:8b",
            info="Granite 3.2 optimized for RAG and tool calling"
        ), gr.update(value="local", interactive=not force_local)
    else:  # openai (normal or ephemeral modes)
        return gr.update(
            choices=[
                ("GPT-4o Mini (Fast & Cheap)", "gpt-4o-mini"),
                ("GPT-4o (Balanced)", "gpt-4o"),
                ("GPT-4 Turbo", "gpt-4-turbo"),
                ("GPT-3.5 Turbo (Fastest)", "gpt-3.5-turbo"),
            ],
            value="gpt-4o-mini",
            info="OpenAI models (API) - Recommended for complex queries"
        ), gr.update(interactive=True)


def create_chat_interface(ui):
    """Create the chat tab interface."""
    from src.ui.utils import get_available_documents, format_document_list, format_doc_dropdown_choices

    with gr.Column():
        with gr.Row():
            with gr.Column(scale=2):
                gr.Markdown("### Chat with Docs")

                with gr.Row():
                    doc_dropdown = gr.Dropdown(
                        choices=format_doc_dropdown_choices(get_available_documents()),
                        value="none",
                        label="Select Document (optional)",
                        info="Auto-injects document title into queries",
                        scale=1,
                    )

                    default_provider = os.getenv("LLM_PROVIDER", "openai")
                    if default_provider == "local":
                        default_model_choices = [
                            ("Granite 3.2 8B (Recommended)", "granite3.2:8b"),
                            ("Llama 3.2 3B (Faster, less accurate)", "llama3.2:3b"),
                        ]
                        default_model = "granite3.2:8b"
                        default_model_info = "Granite 3.2 optimized for RAG and tool calling"
                    else:
                        default_model_choices = [
                            ("GPT-4o Mini (Fast & Cheap)", "gpt-4o-mini"),
                            ("GPT-4o (Balanced)", "gpt-4o"),
                            ("GPT-4 Turbo", "gpt-4-turbo"),
                            ("GPT-3.5 Turbo (Fastest)", "gpt-3.5-turbo"),
                        ]
                        default_model = "gpt-4o-mini"
                        default_model_info = "Select model for chat"

                    provider_dropdown = gr.Dropdown(
                        choices=[
                            ("OpenAI (API)", "openai"),
                            ("Local Ollama", "local"),
                        ],
                        value=default_provider,
                        label="Provider",
                        info="Select LLM provider",
                        scale=0.85,
                    )

                    model_dropdown = gr.Dropdown(
                        choices=default_model_choices,
                        value=default_model,
                        label="Model",
                        info=default_model_info,
                        scale=0.85,
                    )

                    privacy_mode = gr.Radio(
                        choices=[
                            ("Normal", "normal"),
                            ("Ephemeral", "ephemeral"),
                            ("Internal", "internal"),
                            ("Private", "private")
                        ],
                        value="normal",
                        label="Privacy Mode",
                        scale=1.3,
                    )

                chatbot = gr.Chatbot(
                    height=600, show_label=False, avatar_images=(None, None)
                )

                with gr.Row():
                    msg = gr.Textbox(
                        placeholder="Ask about a document...", show_label=False, scale=9
                    )
                    send_btn = gr.Button("Send", scale=1, variant="primary")

                with gr.Row():
                    clear_btn = gr.Button("Clear Conversation")

                # Feedback section - compact single row
                with gr.Row(visible=False) as feedback_row:
                    with gr.Column(scale=1):
                        gr.Markdown("**Rate:**")
                    with gr.Column(scale=6):
                        rating_radio = gr.Radio(
                            choices=[
                                ("★", 1),
                                ("★★", 2),
                                ("★★★", 3),
                                ("★★★★", 4),
                                ("★★★★★", 5),
                            ],
                            label="",
                            show_label=False,
                        )
                    with gr.Column(scale=1):
                        submit_rating_btn = gr.Button(
                            "Submit", variant="primary", size="sm"
                        )

                feedback_status = gr.Textbox(visible=False, show_label=False)

                # Save as Note section
                with gr.Accordion("Save as Note", open=False, visible=False) as save_note_accordion:
                    note_title = gr.Textbox(label="Title", placeholder="Note title...")
                    note_tags = gr.Textbox(
                        label="Tags (comma-separated)",
                        placeholder="e.g. odyssey, themes, characters",
                    )
                    note_content = gr.Textbox(
                        label="Content (editable)",
                        lines=8,
                        placeholder="Response content will appear here for editing...",
                    )
                    note_sources_display = gr.Markdown(
                        value="*No source references*", label="Sources"
                    )
                    with gr.Row():
                        save_note_btn = gr.Button("Save Note", variant="primary", size="sm")
                        note_save_status = gr.Textbox(
                            visible=False, show_label=False, interactive=False
                        )

                with gr.Accordion("Tips", open=False):
                    gr.Markdown(
                        """
                        - **Doc Selection**: Use dropdown or mention document title in your query
                        - **Search Examples**: "Find passages about X", "What does the author say about Y?"
                        - **Section Context**: Ask about specific sections or broad themes
                        - **Hybrid Search**: Uses both keyword matching (BM25) and semantic search
                        - **Rate Responses**: Help improve quality by rating answers

                        **Local LLM (Llama 3.1 8B) Notes:**
                        - Works well for: Questions, summaries, multi-document comparisons
                        - Good function calling ability, handles complex queries
                        - Slower than cloud APIs (especially first request after model load)
                        - Requires 10GB Docker memory, uses 8B parameter model
                        """
                    )

            with gr.Column(scale=1, min_width=300):
                gr.Markdown("### Library")

                doc_list = gr.Dataframe(
                    headers=["Slug", "Title", "Author", "Chunks", "Added"],
                    datatype=["str", "str", "str", "number", "str"],
                    interactive=False,
                    wrap=True,
                    column_widths=["15%", "25%", "20%", "12%", "28%"],
                    max_height=800,
                )

        # Event handlers - wrap to pass ui
        submit_outputs = [
            chatbot, msg, feedback_row,
            save_note_accordion, note_title, note_content, note_sources_display,
        ]

        async def handle_submit(msg_text, history, doc_sel, provider_sel, model_sel, privacy):
            async for result in respond(
                msg_text, history, doc_sel, provider_sel, model_sel, privacy, ui
            ):
                yield result

        def handle_rating(rating, history):
            status = submit_feedback(rating, history)
            return status, gr.update(visible=False)

        # Update model dropdown when provider or privacy mode changes
        provider_dropdown.change(
            update_model_choices,
            [provider_dropdown, privacy_mode],
            [model_dropdown, provider_dropdown]
        )
        privacy_mode.change(
            update_model_choices,
            [provider_dropdown, privacy_mode],
            [model_dropdown, provider_dropdown]
        )

        msg.submit(
            handle_submit,
            [msg, chatbot, doc_dropdown, provider_dropdown, model_dropdown, privacy_mode],
            submit_outputs,
        )
        send_btn.click(
            handle_submit,
            [msg, chatbot, doc_dropdown, provider_dropdown, model_dropdown, privacy_mode],
            submit_outputs,
        )
        clear_btn.click(
            lambda: ([], gr.update(visible=False), gr.update(visible=False)),
            None,
            [chatbot, feedback_row, save_note_accordion],
        )

        submit_rating_btn.click(
            handle_rating, [rating_radio, chatbot], [feedback_status, feedback_row]
        )

        # Save as Note handler
        save_note_btn.click(
            save_as_note,
            [note_title, note_content, note_tags, chatbot],
            [note_save_status],
        )

        # Load document list on page load
        def load_doc_list():
            return format_document_list(get_available_documents())

    return (doc_dropdown, doc_list, load_doc_list)
