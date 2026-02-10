import asyncio
import random
from collections import defaultdict
from openai import AsyncOpenAI
import tiktoken


# Templates
SECTION_TEMPLATE = """Summarize this {unit} into 1–2 concise paragraphs.
Focus on: {focus}
Avoid bullet points. Do not mention chunking.

{unit_label}:
{text}"""

OVERALL_TEMPLATE = """Summarize these {unit} summaries into 2–3 cohesive paragraphs.
Focus on: {focus}

{unit_label} summaries:
{joined}"""

# Doc-type-specific parameters
DOC_PARAMS = {
    'book': {
        'unit': 'chapter',
        'unit_label': 'Chapter text',
        'section_focus': 'key events, themes, and character actions',
        'overall_focus': 'major themes, central characters, and overall narrative arc'
    },
    'conversation': {
        'unit': 'conversation segment',
        'unit_label': 'Conversation excerpt',
        'section_focus': 'key discussion points, decisions made, and action items',
        'overall_focus': 'main topics discussed, key decisions and outcomes, action items'
    },
    'script': {
        'unit': 'scene',
        'unit_label': 'Scene text',
        'section_focus': 'what happens, character interactions, and plot progression',
        'overall_focus': 'story arc, key characters, and major plot points'
    },
    'tech_doc': {
        'unit': 'section',
        'unit_label': 'Documentation text',
        'section_focus': 'key concepts, features, and technical details',
        'overall_focus': 'main topics and concepts, key features, important technical details'
    },
    'report': {
        'unit': 'section',
        'unit_label': 'Report section',
        'section_focus': 'key findings, data points, and insights',
        'overall_focus': 'main findings and insights, key data points, conclusions and recommendations'
    }
}


class SummaryGenerator:
    def __init__(self, doc_type: str = 'book', provider: str = None):
        from src.llm.config import LLMConfig
        self.config = LLMConfig.from_env()
        self.provider = provider or self.config.default_provider
        
        if self.provider == "local" or self.config.privacy_mode:
            # Ensure /v1 for OpenAI-compatible API
            base_url = self.config.ollama_base_url
            if "/v1" not in base_url:
                base_url = base_url.rstrip("/") + "/v1"
            self.client = AsyncOpenAI(base_url=base_url, api_key="ollama")
            self.model = self.config.ollama_model
        else:
            self.client = AsyncOpenAI(api_key=self.config.openai_api_key)
            self.model = "gpt-4o-mini"

        # RATE LIMIT FIX: Reduce concurrency from 4 -> 2 to avoid TPM 429 errors
        self.semaphore = asyncio.Semaphore(2)
        self.doc_type = doc_type
        params = DOC_PARAMS.get(doc_type, DOC_PARAMS['book'])

        # Build prompts from template
        self.section_prompt = SECTION_TEMPLATE.format(
            unit=params['unit'],
            unit_label=params['unit_label'],
            focus=params['section_focus'],
            text='{text}'
        )
        self.overall_prompt = OVERALL_TEMPLATE.format(
            unit=params['unit'],
            unit_label=params['unit_label'],
            focus=params['overall_focus'],
            joined='{joined}'
        )
        self.enc = tiktoken.get_encoding("cl100k_base")
        self.max_tokens = 100000  # Safe limit well below 200k TPM

    def split_text_into_batches(self, text: str, max_tokens: int = None):
        """Split text into batches that fit within token limit."""
        if max_tokens is None:
            max_tokens = self.max_tokens

        tokens = self.enc.encode(text)

        if len(tokens) <= max_tokens:
            return [text]

        # Split into roughly equal batches
        num_batches = (len(tokens) // max_tokens) + 1
        batch_size = len(tokens) // num_batches

        batches = []
        for i in range(0, len(tokens), batch_size):
            batch_tokens = tokens[i:i + batch_size]
            batch_text = self.enc.decode(batch_tokens)
            batches.append(batch_text)

        return batches

    async def summarize_text_batch(self, text: str, batch_num: int = None):
        """Summarize a batch of text with robust retry logic."""
        max_retries = 5
        retry_delay = 5 # seconds

        # Jitter to desynchronize requests
        await asyncio.sleep(random.uniform(0.1, 0.5))

        for attempt in range(max_retries):
            async with self.semaphore:
                try:
                    prompt = self.section_prompt.format(text=text)
                    resp = await self.client.chat.completions.create(
                        model=self.model,  # fast + cheap
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.3,
                        timeout=120.0 # Prevent timeouts on large contexts
                    )
                    
                    # Safety cooldown
                    await asyncio.sleep(0.2)
                    
                    return str(resp.choices[0].message.content).strip()

                except Exception as e:
                    error_msg = str(e).lower()
                    if ("rate_limit" in error_msg or "429" in error_msg) and attempt < max_retries - 1:
                        wait_time = retry_delay * (2 ** attempt)
                        print(f"[Summary] Rate limit hit (Attempt {attempt+1}/{max_retries}). Retrying in {wait_time}s...")
                        await asyncio.sleep(wait_time)
                        continue
                    
                    if attempt == max_retries - 1:
                        print(f"[Summary Error] Failed after {max_retries} attempts: {e}")
                        raise e # Fail hard if we can't summarize
                    
                    # Other errors (network, timeout) also get retried
                    print(f"[Summary Warning] Transient error: {e}. Retrying...")
                    await asyncio.sleep(2)

    async def summarize_chapter(self, text, sid):
        """Summarize a single chapter, handling large texts by batching."""
        # Check if text is too large
        batches = self.split_text_into_batches(text)

        if len(batches) == 1:
            # Small enough to summarize directly
            summary = await self.summarize_text_batch(text)
        else:
            # Need to batch: summarize each batch, then combine summaries
            print(f"Chapter {sid} is large ({len(batches)} batches), summarizing in parts...")
            batch_summaries = await asyncio.gather(*[
                self.summarize_text_batch(batch, i)
                for i, batch in enumerate(batches)
            ])

            # Combine batch summaries into final summary
            combined_text = "\n\n".join([
                f"Part {i+1}: {summary}"
                for i, summary in enumerate(batch_summaries)
            ])

            # Summarize the combined summaries
            params = DOC_PARAMS.get(self.doc_type, DOC_PARAMS['book'])
            
            # Simple retry wrapper for final synthesis call
            for attempt in range(3):
                try:
                    async with self.semaphore:
                        prompt = f"""
These are summaries of different parts of the same {params['unit']}.
Combine them into 1-2 cohesive paragraphs that capture the key points.

{combined_text}
"""
                        resp = await self.client.chat.completions.create(
                            model=self.model,
                            messages=[{"role": "user", "content": prompt}],
                            temperature=0.3,
                        )
                        summary = str(resp.choices[0].message.content).strip()
                        break
                except Exception as e:
                    if attempt == 2: 
                        raise e
                    await asyncio.sleep(5)

        return {
            "chapter_id": sid,
            "summary": summary,
        }

    async def summarize_book(self, chapter_summaries):
        """Synthesize whole-document summary from section summaries."""
        params = DOC_PARAMS.get(self.doc_type, DOC_PARAMS['book'])
        unit = params['unit'].capitalize()

        joined = "\n\n".join(
            [f"{unit} {c['chapter_id']}: {c['summary']}" for c in chapter_summaries]
        )

        # Check if combined summaries are too large
        batches = self.split_text_into_batches(joined, max_tokens=25000)  # Leave room for prompt

        if len(batches) == 1:
            # Small enough to summarize directly
            prompt = self.overall_prompt.format(joined=joined)
            
            # Use summarize_text_batch internal logic via direct call or just wrap here
            # Wrapping logic for safety
            for attempt in range(3):
                try:
                    async with self.semaphore:
                        resp = await self.client.chat.completions.create(
                            model=self.model,  # Use mini for cost efficiency
                            messages=[{"role": "user", "content": prompt}],
                            temperature=0.3,
                        )
                        return str(resp.choices[0].message.content).strip()
                except Exception as e:
                    if attempt == 2: 
                        raise e
                    await asyncio.sleep(5)
        else:
            # Document is large, batch the summaries
            print(f"Document is large ({len(batches)} batches), summarizing in parts...")
            batch_summaries = []

            for i, batch in enumerate(batches):
                prompt = self.overall_prompt.format(joined=batch)
                async with self.semaphore:
                    # Simple call - in a real scenario we'd use retry here too
                    # But for now, relying on lower concurrency to save us
                    resp = await self.client.chat.completions.create(
                        model=self.model,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.3,
                    )
                    batch_summaries.append(str(resp.choices[0].message.content).strip())

            # Combine batch summaries into final summary
            if len(batch_summaries) == 1:
                return batch_summaries[0]

            combined = "\n\n".join([
                f"Part {i+1}: {summary}"
                for i, summary in enumerate(batch_summaries)
            ])

            # Final synthesis
            async with self.semaphore:
                final_prompt = f"""
These are summaries of different parts of the same document.
Combine them into 2-3 cohesive paragraphs that capture the document's key themes and content.

{combined}
"""
                resp = await self.client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": final_prompt}],
                    temperature=0.3,
                )
                return str(resp.choices[0].message.content).strip()

    async def summarize_hierarchy(self, chunks):
        """
        Summarize book hierarchy.
        Returns (chapter_summaries, book_summary).
        """
        print(f"[SUMMARY] Starting summarization for {len(chunks)} chunks...")

        # Group chunks by chapter_id (sid)
        section_map = defaultdict(list)
        for c in chunks:
            # Robust parsing: sid is the 3rd-to-last part of the ID
            # ID format: {slug}_{sid}_{cid}_{hash}
            # e.g. "mini_iliad_01_001_abc123" -> "01"
            parts = c["id"].split("_")
            try:
                sid = int(parts[-3])
            except (ValueError, IndexError):
                # Fallback to 1 if we can't parse or format is different
                sid = 1
            section_map[sid].append(c["text"])

        # Join chapter text
        chapters = {sid: "\n".join(texts) for sid, texts in section_map.items()}

        print(f"[SUMMARY] Grouped into {len(chapters)} sections. Summarizing sections...")

        # Summarize chapters in parallel
        tasks = [self.summarize_chapter(text, sid) for sid, text in chapters.items()]
        chapter_summaries = await asyncio.gather(*tasks)

        print("[SUMMARY] Section summaries complete. Generating final summary...")

        # Summarize entire book
        book_summary = await self.summarize_book(chapter_summaries)

        print("[SUMMARY] Final summary complete!")

        return chapter_summaries, book_summary