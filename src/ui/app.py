"""
Main Gradio application for Doc Mate.
"""

import gradio as gr
import asyncio
import os
import threading
from src.mcp_client.agent import BookMateAgent
from src.ui.chat import create_chat_interface
from src.ui.ingest import create_ingest_interface
from src.ui.monitoring import create_monitoring_interface
from src.flows.document_query import preload_retriever

# NOTE: Phoenix tracing is initialized on-demand in BookMateUI.__init__
# to respect ephemeral mode flags. Do NOT initialize here at module load.


class DocMateUI:
    """Main UI controller managing the MCP agent."""

    def __init__(self):
        # We no longer keep a persistent agent to avoid asyncio task context issues
        self.provider = "openai"  # Default provider
        self.model = "gpt-4o-mini"  # Default model
        self.privacy_mode = "normal"  # Default: normal mode

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

        return changed, old_ephemeral, new_ephemeral

    async def chat(
        self, message: str, history: list, selected_doc: str = None
    ) -> tuple[str, str]:
        """
        Handle chat messages with the agent.

        Args:
            message: User message
            history: Gradio chat history format
            selected_doc: Selected document slug (optional)

        Returns:
            (agent_response, query_id)
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
                    with store.conn.cursor() as cur:
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

            response, _, query_id = await agent.chat(message, conversation_history)
            return response, query_id

        except Exception as e:
            print(f"Chat error: {e}")
            return f"Error: {str(e)}. Please try again.", None
        
        finally:
            # Always close the agent to clean up subprocesses
            await agent.close()

    async def cleanup(self):
        """Clean up agent resources."""
        # No persistent agent to clean up anymore
        pass


def create_app():
    """Create the main Gradio application."""
    from src.ui.utils import get_available_documents, format_document_list

    # Preload retriever in background to avoid delay on first query
    threading.Thread(target=preload_retriever, daemon=True).start()

    ui = DocMateUI()

    with gr.Blocks(title="Doc Mate", theme=gr.themes.Base()) as app:
        gr.Markdown("# Doc Mate - AI Document Assistant")

        with gr.Tabs() as tabs:
            # Tab 1: Chat Interface
            with gr.Tab("Chat", id=0):
                dropdown, doc_list, load_doc_list = create_chat_interface(ui)

            # Tab 2: Add New Document
            with gr.Tab("Add Document", id=1):
                ingest_doc_list = create_ingest_interface()

            # Tab 3: Monitoring
            with gr.Tab("Monitoring", id=2):
                create_monitoring_interface()

        # Auto-refresh document lists when switching tabs
        def refresh_on_tab_change(evt: gr.SelectData):
            # Always fetch fresh data from database (source of truth)
            docs = get_available_documents()
            new_list = format_document_list(docs)
            # Show only titles in dropdown, not slugs
            new_choices = [("Select a doc...", "none")] + [
                (f"{title}", slug) for slug, title, _, _, _ in docs
            ]

            print(
                f"[DEBUG] Tab switched to: {evt.value}, refreshing with {len(docs)} documents"
            )

            if evt.value == 0 or evt.index == 0:
                # Switching to Chat tab - refresh chat document list and dropdown
                return new_list, gr.update(choices=new_choices), gr.update()
            elif evt.value == 1 or evt.index == 1:
                # Switching to Add Document tab - refresh ingest document list
                return gr.update(), gr.update(), new_list

            # Refresh both to be safe
            return new_list, gr.update(choices=new_choices), new_list

        tabs.select(
            refresh_on_tab_change, None, [doc_list, dropdown, ingest_doc_list]
        )

        # Load document lists on startup
        def load_ingest_list():
            return format_document_list(get_available_documents())

        app.load(load_doc_list, None, doc_list)
        app.load(load_ingest_list, None, ingest_doc_list)

    return app


if __name__ == "__main__":
    app = create_app()
    app.launch(server_name="0.0.0.0", server_port=7860, share=False)
