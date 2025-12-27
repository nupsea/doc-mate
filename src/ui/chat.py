"""
Chat interface component.
"""

import gradio as gr
from src.ui.utils import get_available_books, format_book_list
from src.monitoring.metrics import metrics_collector


# Store query_id for each chat turn (message index -> query_id)
query_id_map = {}


async def respond(message, chat_history, selected_book, selected_provider, selected_model, privacy_mode, ui):
    """Handle chat interactions."""
    if not message.strip():
        yield chat_history, message, gr.update(visible=False)
        return

    # Update provider/model if changed (with proper cleanup)
    await ui.set_provider_and_model(selected_provider, selected_model, privacy_mode)

    # Add user message with loading indicator for bot
    chat_history.append([message, "Thinking..."])

    # Keep message in textbox during processing
    yield chat_history, message, gr.update(visible=False)

    # Get bot response
    bot_response, query_id = await ui.chat(message, chat_history[:-1], selected_book)

    # Update with actual response
    chat_history[-1][1] = bot_response

    # Store query_id for this interaction
    if query_id:
        query_id_map[len(chat_history) - 1] = query_id

    # Clear textbox and show feedback buttons
    yield chat_history, "", gr.update(visible=True, value=None)


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
            choices=[("Llama 3.1 8B (Local)", "llama3.1:8b")],
            value="llama3.1:8b",
            info="Local Ollama model - good for comparisons"
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

    with gr.Column():
        with gr.Row():
            with gr.Column(scale=2):
                gr.Markdown("### Chat with Docs")

                with gr.Row():
                    book_dropdown = gr.Dropdown(
                        choices=[("Select a doc...", "none")]
                        + [
                            (f"{title}", slug)
                            for slug, title, _, _, _ in get_available_books()
                        ],
                        value="none",
                        label="Select Doc (optional)",
                        info="Auto-injects document title into queries",
                        scale=1,
                    )

                    provider_dropdown = gr.Dropdown(
                        choices=[
                            ("OpenAI (API)", "openai"),
                            ("Local Ollama", "local"),
                        ],
                        value="openai",
                        label="Provider",
                        info="Select LLM provider",
                        scale=0.85,
                    )

                    model_dropdown = gr.Dropdown(
                        choices=[
                            ("GPT-4o Mini (Fast & Cheap)", "gpt-4o-mini"),
                            ("GPT-4o (Balanced)", "gpt-4o"),
                            ("GPT-4 Turbo", "gpt-4-turbo"),
                            ("GPT-3.5 Turbo (Fastest)", "gpt-3.5-turbo"),
                        ],
                        value="gpt-4o-mini",
                        label="Model",
                        info="Select model for chat",
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

                with gr.Accordion("💡 Tips", open=False):
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

                book_list = gr.Dataframe(
                    headers=["Slug", "Title", "Author", "Chunks", "Added"],
                    datatype=["str", "str", "str", "number", "str"],
                    interactive=False,
                    wrap=True,
                    column_widths=["15%", "25%", "20%", "12%", "28%"],
                    max_height=800,
                )

        # Event handlers - wrap to pass ui
        async def handle_submit(msg_text, history, book_sel, provider_sel, model_sel, privacy):
            async for result_history, result_msg, feedback_update in respond(
                msg_text, history, book_sel, provider_sel, model_sel, privacy, ui
            ):
                yield result_history, result_msg, feedback_update

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
            [msg, chatbot, book_dropdown, provider_dropdown, model_dropdown, privacy_mode],
            [chatbot, msg, feedback_row]
        )
        send_btn.click(
            handle_submit,
            [msg, chatbot, book_dropdown, provider_dropdown, model_dropdown, privacy_mode],
            [chatbot, msg, feedback_row]
        )
        clear_btn.click(
            lambda: ([], gr.update(visible=False)), None, [chatbot, feedback_row]
        )

        submit_rating_btn.click(
            handle_rating, [rating_radio, chatbot], [feedback_status, feedback_row]
        )

        # Load book list on page load
        def load_book_list():
            return format_book_list(get_available_books())

    return book_dropdown, book_list, load_book_list
