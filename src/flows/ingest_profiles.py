"""
Ingestion profiles -- speed/quality/cost tradeoffs for document ingestion.

Each profile controls which LLM provider handles summaries and graph extraction,
graph coverage level (full vs sampled vs skip), and concurrency settings.
"""

import math
from dataclasses import dataclass


@dataclass
class IngestProfile:
    """Configuration for a single ingestion profile."""

    name: str
    label: str
    description: str
    summary_provider: str       # "openai" | "local"
    graph_provider: str         # "openai" | "local" | "skip"
    graph_coverage: str         # "full" | "sampled" | "skip"
    graph_sample_pct: float     # 1.0 = all chunks, 0.5 = half
    summary_sampling: str       # "auto" | "aggressive"
    summary_semaphore: int      # concurrency for SummaryGenerator
    graph_semaphore: int        # concurrency for EntityExtractor
    graph_batch_size: int       # chunks per extraction batch

    def estimate_seconds(self, num_chunks: int, doc_type: str) -> tuple:
        """Return (min_seconds, max_seconds) estimate for this profile + chunk count."""

        # --- Summary estimate ---
        if self.summary_sampling == "aggressive":
            summary_chunks = min(num_chunks, 20)
        elif doc_type == "conversation" and num_chunks >= 100:
            summary_chunks = 30  # auto sampling for long conversations
        else:
            summary_chunks = num_chunks

        summary_sections = max(1, summary_chunks // 7)

        if self.summary_provider == "openai":
            summary_rounds = math.ceil(summary_sections / self.summary_semaphore)
            summary_min = summary_rounds * 5
            summary_max = summary_rounds * 15
        else:
            # Local Ollama: effectively serial (OLLAMA_NUM_PARALLEL=1)
            summary_min = summary_sections * 15
            summary_max = summary_sections * 40

        # --- Graph estimate ---
        if self.graph_coverage == "skip" or self.graph_provider == "skip":
            graph_min, graph_max = 0, 0
        else:
            graph_chunk_count = int(num_chunks * self.graph_sample_pct)
            entity_batches = math.ceil(graph_chunk_count / self.graph_batch_size)
            episode_bs = max(self.graph_batch_size // 2, 2)
            episode_batches = math.ceil(graph_chunk_count / episode_bs) if doc_type == "conversation" else 0
            total_batches = entity_batches + episode_batches

            if self.graph_provider == "openai":
                rounds = math.ceil(total_batches / self.graph_semaphore)
                graph_min = rounds * 10
                graph_max = rounds * 20
            else:
                # Local: serial processing
                graph_min = total_batches * 20
                graph_max = total_batches * 45

        # Search indexing is fast and parallel with LLM work
        search_min, search_max = 30, 60

        # Summaries + Graph run in parallel (asyncio.gather), total = max
        llm_min = max(summary_min, graph_min)
        llm_max = max(summary_max, graph_max)

        total_min = max(llm_min, search_min) + 15   # parsing/validation overhead
        total_max = max(llm_max, search_max) + 30

        return total_min, total_max


PROFILES = {
    "openai_full": IngestProfile(
        name="openai_full",
        label="Full Quality (OpenAI)",
        description="All processing via OpenAI. Best quality, highest cost.",
        summary_provider="openai",
        graph_provider="openai",
        graph_coverage="full",
        graph_sample_pct=1.0,
        summary_sampling="auto",
        summary_semaphore=2,
        graph_semaphore=2,
        graph_batch_size=8,
    ),
    "openai_fast": IngestProfile(
        name="openai_fast",
        label="Fast (OpenAI, sampled graph)",
        description="OpenAI for all LLM work, but graph extraction samples 50% of chunks. ~2x faster graph phase.",
        summary_provider="openai",
        graph_provider="openai",
        graph_coverage="sampled",
        graph_sample_pct=0.5,
        summary_sampling="auto",
        summary_semaphore=2,
        graph_semaphore=2,
        graph_batch_size=8,
    ),
    "hybrid": IngestProfile(
        name="hybrid",
        label="Hybrid (Local summaries, OpenAI graph)",
        description="Summaries via local Granite, graph extraction via OpenAI. Saves cost on summaries while keeping graph quality.",
        summary_provider="local",
        graph_provider="openai",
        graph_coverage="full",
        graph_sample_pct=1.0,
        summary_sampling="auto",
        summary_semaphore=2,
        graph_semaphore=2,
        graph_batch_size=8,
    ),
    "local_full": IngestProfile(
        name="local_full",
        label="Fully Local (Granite)",
        description="Everything via local Granite 3.2 8B. Zero API cost, no data leaves your machine. Graph sampled at 50% to keep time reasonable.",
        summary_provider="local",
        graph_provider="local",
        graph_coverage="sampled",
        graph_sample_pct=0.5,
        summary_sampling="auto",
        summary_semaphore=2,
        graph_semaphore=2,
        graph_batch_size=4,
    ),
    "search_only": IngestProfile(
        name="search_only",
        label="Search Only (skip graph)",
        description="Summaries via OpenAI, skip graph extraction entirely. Fastest option -- search works, but no entity/relationship knowledge.",
        summary_provider="openai",
        graph_provider="skip",
        graph_coverage="skip",
        graph_sample_pct=0.0,
        summary_sampling="auto",
        summary_semaphore=2,
        graph_semaphore=2,
        graph_batch_size=8,
    ),
}

DEFAULT_PROFILE = "openai_full"


def check_ollama_available() -> bool:
    """Quick health check: can we reach the Ollama server?"""
    import requests
    from src.llm.config import LLMConfig

    config = LLMConfig.from_env()
    try:
        base = config.ollama_base_url.rstrip("/")
        resp = requests.get(f"{base}/api/tags", timeout=3)
        return resp.status_code == 200
    except Exception:
        return False
