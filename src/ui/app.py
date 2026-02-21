"""
Main Gradio application for Doc Mate.
"""

import gradio as gr
import os
import threading
from src.mcp_client.agent import BookMateAgent
from src.ui.chat import create_chat_interface
from src.ui.ingest import create_ingest_interface, _format_job_banner
from src.ui.notes import create_notes_interface
from src.ui.monitoring import create_monitoring_interface
from src.flows.document_query import preload_retriever
from src.content.db import DatabaseManager

# NOTE: Phoenix tracing is initialized on-demand in BookMateUI.__init__
# to respect ephemeral mode flags. Do NOT initialize here at module load.


class DocMateUI:
    """Main UI controller managing the MCP agent."""

    def __init__(self):
        # We no longer keep a persistent agent to avoid asyncio task context issues
        self.provider = "openai"  # Default provider
        self.model = "gpt-4o-mini"  # Default model
        self.privacy_mode = "normal"  # Default: normal mode
        self._fallback_shown = False  # Show fallback notice only once

    async def set_provider_and_model(self, provider: str, model: str, privacy_mode: str):
        """Set the LLM provider, model, and privacy mode.

        Returns:
            tuple: (changed, was_ephemeral, is_ephemeral)
                - changed: True if any settings changed
                - was_ephemeral: True if previous mode was ephemeral
                - is_ephemeral: True if new mode is ephemeral
        """
        changed = (provider != self.provider or model != self.model or privacy_mode != self.privacy_mode)

        # Determine if old/new modes are ephemeral (for conversation history handling)
        old_ephemeral = self.privacy_mode in ["ephemeral", "private"]
        new_ephemeral = privacy_mode in ["ephemeral", "private"]

        if changed:
            print(f"Changing from {self.provider}/{self.model} to {provider}/{model} (privacy={privacy_mode})")
            self.provider = provider
            self.model = model
            self.privacy_mode = privacy_mode
            self._fallback_shown = False

        return changed, old_ephemeral, new_ephemeral

    async def chat(
        self, message: str, history: list, selected_doc: str = None
    ) -> tuple[str, str, list]:
        """
        Handle chat messages with the agent.

        Args:
            message: User message
            history: Gradio chat history format
            selected_doc: Selected document slug (optional)

        Returns:
            (agent_response, query_id, source_refs)
        """
        # Parse privacy mode into flags
        ephemeral = self.privacy_mode in ["ephemeral", "private"]
        internal_mode = self.privacy_mode in ["internal", "private"]

        # Set environment for ModelRouter (per request)
        if internal_mode:
            os.environ["LLM_PROVIDER"] = "local"
        else:
            os.environ["LLM_PROVIDER"] = self.provider

        # Initialize agent per request to ensure thread safety
        agent = BookMateAgent(
            provider=self.provider,
            model=self.model,
            ephemeral=ephemeral,
            internal_mode=internal_mode
        )

        try:
            await agent.connect_to_mcp_server()

            # Auto-inject document title if selected
            print(f"\n[UI] Original message: {message}")
            print(f"[UI] Selected document slug from dropdown: {selected_doc}")

            if selected_doc and selected_doc != "none":
                # Get document title from slug
                from src.content.store import PgresStore

                try:
                    store = PgresStore()
                    with store._get_conn() as conn:
                        with conn.cursor() as cur:
                            cur.execute(
                                "SELECT title FROM documents WHERE slug = %s", (selected_doc,)
                            )
                            result = cur.fetchone()
                            if result:
                                doc_title = result[0]
                                print(
                                    f"[UI] Found document title for slug '{selected_doc}': {doc_title}"
                                )
                                # Only inject if not already mentioned
                                if doc_title.lower() not in message.lower():
                                    message = f"{message} (for the document '{doc_title}')"
                                    print(f"[UI] Injected title into message: {message}")
                                else:
                                    print("[UI] Title already in message, not injecting")
                except Exception as e:
                    print(f"[WARN] Could not get document title: {e}")
            else:
                print("[UI] No document selected from dropdown")

            # Convert Gradio history to OpenAI format
            conversation_history = []
            for user_msg, bot_msg in history:
                conversation_history.append({"role": "user", "content": user_msg})
                if bot_msg:
                    conversation_history.append({"role": "assistant", "content": bot_msg})

            doc_slug = selected_doc if selected_doc and selected_doc != "none" else None
            response, _, query_id, source_refs = await agent.chat(message, conversation_history, selected_doc=doc_slug)

            if agent.fallback_notice and not self._fallback_shown:
                response = f"> **Note:** {agent.fallback_notice}\n\n{response}"
                self._fallback_shown = True

            return response, query_id, source_refs

        except Exception as e:
            print(f"Chat error: {e}")
            return f"Error: {str(e)}. Please try again.", None, []
        
        finally:
            # Always close the agent to clean up subprocesses
            await agent.close()

    async def cleanup(self):
        """Clean up agent resources."""
        # No persistent agent to clean up anymore
        pass


def create_app():
    """Create the main Gradio application."""
    # Initialize DB Pool
    DatabaseManager.get_pool()
    
    from src.ui.utils import get_available_documents, format_document_list, format_doc_dropdown_choices

    # Preload retriever in background to avoid delay on first query
    threading.Thread(target=preload_retriever, daemon=True).start()

    ui = DocMateUI()

    with gr.Blocks(title="Doc Mate", theme=gr.themes.Base()) as app:
        gr.Markdown("# Doc Mate - AI Document Assistant")

        with gr.Tabs() as tabs:
            # Tab 0: Chat Interface
            with gr.Tab("Chat", id=0):
                dropdown, doc_list, load_doc_list = create_chat_interface(ui)

            # Tab 1: Notes
            with gr.Tab("Notes", id=1):
                notes_table, notes_tag_filter, load_notes = create_notes_interface()

            # Tab 2: Add New Document
            with gr.Tab("Add Document", id=2):
                ingest_doc_list, ingest_job_banner = create_ingest_interface()

            # Tab 3: Monitoring
            with gr.Tab("Monitoring", id=3):
                create_monitoring_interface()

        # Auto-refresh document lists when switching tabs
        def refresh_on_tab_change(evt: gr.SelectData):
            # Always fetch fresh data from database (source of truth)
            docs = get_available_documents()
            new_list = format_document_list(docs)
            new_choices = format_doc_dropdown_choices(docs)

            # Refresh ingest job banner
            from src.content.store import PgresStore
            try:
                banner = _format_job_banner(PgresStore().get_latest_ingest_job())
            except Exception:
                banner = ""

            # Refresh notes list
            from src.ui.notes import _load_note_list, _get_all_tags
            note_rows = _load_note_list()
            note_tags = _get_all_tags()

            print(
                f"[DEBUG] Tab switched to: {evt.value}, refreshing with {len(docs)} documents"
            )

            if evt.index == 0:
                # Chat tab
                return (new_list, gr.update(choices=new_choices),
                        gr.update(), gr.update(), gr.update(), banner)
            elif evt.index == 1:
                # Notes tab
                return (gr.update(), gr.update(),
                        note_rows, gr.update(choices=note_tags), gr.update(), banner)
            elif evt.index == 2:
                # Add Document tab
                return (gr.update(), gr.update(),
                        gr.update(), gr.update(), new_list, banner)

            # Default: refresh all
            return (new_list, gr.update(choices=new_choices),
                    note_rows, gr.update(choices=note_tags), new_list, banner)

        tabs.select(
            refresh_on_tab_change, None,
            [doc_list, dropdown, notes_table, notes_tag_filter, ingest_doc_list, ingest_job_banner]
        )

        # Load document lists on startup
        def load_ingest_list():
            return format_document_list(get_available_documents())

        def load_job_banner():
            from src.content.store import PgresStore
            try:
                return _format_job_banner(PgresStore().get_latest_ingest_job())
            except Exception:
                return ""

        def load_notes_initial():
            rows, tag_update = load_notes()
            return rows, tag_update

        app.load(load_doc_list, None, doc_list)
        app.load(load_notes_initial, None, [notes_table, notes_tag_filter])
        app.load(load_ingest_list, None, ingest_doc_list)
        app.load(load_job_banner, None, ingest_job_banner)

    return app


if __name__ == "__main__":
    app = create_app()
    app.launch(server_name="0.0.0.0", server_port=7860, share=False)
