import asyncio
import os
import random
from typing import List, Dict, Any
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from src.graph.schemas import Entity, Relationship, Episode, DOC_TYPE_SCHEMAS

# --- Pydantic Models for Structured Output ---

class ExtractedEntity(BaseModel):
    """Temporary model for LLM output structure"""
    name: str = Field(description="Canonical name of the entity")
    entity_type: str = Field(description="Type of the entity")
    description: str = Field(description="Brief description of the entity's role or context")
    source_chunk_ids: List[str] = Field(description="List of chunk IDs where this entity appears")

class ExtractedRelationship(BaseModel):
    """Temporary model for LLM output structure"""
    source_entity: str = Field(description="Name of the source entity")
    target_entity: str = Field(description="Name of the target entity")
    relation_type: str = Field(description="Type of the relationship")
    description: str = Field(description="Context or details about the relationship")
    source_chunk_ids: List[str] = Field(description="List of chunk IDs where this relationship appears")

class ExtractionResult(BaseModel):
    """Container for batch extraction results"""
    entities: List[ExtractedEntity] = Field(default_factory=list)
    relationships: List[ExtractedRelationship] = Field(default_factory=list)

class ExtractedEpisode(BaseModel):
    """Temporary model for Episode output structure"""
    speaker: str = Field(description="Name of the speaker")
    topic: str = Field(description="Main topic discussed")
    stance: str = Field(description="Speaker's stance (e.g., supportive, critical, neutral)")
    summary: str = Field(description="Brief summary of the episode")
    mentioned_entities: List[str] = Field(default_factory=list, description="List of entity names mentioned")
    source_chunk_ids: List[str] = Field(description="List of chunk IDs for this episode")

class EpisodeResult(BaseModel):
    """Container for batch episode results"""
    episodes: List[ExtractedEpisode] = Field(default_factory=list)

# --- Extractor Class ---

class EntityExtractor:
    def __init__(self, provider: str = "openai", model_name: str = "gpt-4o-mini", batch_size: int = 8):
        self.batch_size = batch_size
        # OPTIMIZED FIX: Concurrency=2 with batch_size=8 pushes ~160k TPM.
        # This maximizes speed while staying under the 200k TPM limit.
        self.semaphore = asyncio.Semaphore(2) 
        self.provider = provider
        
        # Initialize Model with higher timeout to prevent "Connection error" on large docs
        if provider == "openai":
            self.llm = ChatOpenAI(
                model=model_name, 
                temperature=0.0,
                request_timeout=120.0 # Allow 2 mins for extraction
            )
        else:
            base_url = os.getenv("OLLAMA_HOST_URL", "http://host.docker.internal:11434")
            self.llm = ChatOllama(
                model=model_name, 
                temperature=0.0, 
                base_url=base_url,
                timeout=120.0
            )

        # 1. Entity Extraction Chain
        entity_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an expert information extraction system. Extract entities and relationships based on the provided schema. Be thorough: find all prominent entities and their connections."),
            ("user", """
Context: Extracting from a {doc_type}. 
Focus Entity Types: {entity_types}
Focus Relationship Types: {rel_types}
Guidance: {focus}

Text Content (with IDs):
{text_content}
""")
        ])
        self.entity_chain = entity_prompt | self.llm.with_structured_output(ExtractionResult)

        # 2. Episode Extraction Chain
        episode_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an expert transcript analyzer. Identify distinct episodes where speakers take stances or discuss topics."),
            ("user", """
Analyze the following conversation text chunks.
Identify episodes/turns including speaker, stance, topic, and summary.

Text Content (with IDs):
{text_content}
""")
        ])
        self.episode_chain = episode_prompt | self.llm.with_structured_output(EpisodeResult)

    def _is_social_chat(self, chunks: List[Dict[str, Any]]) -> bool:
        """Heuristic to detect if a conversation is casual/social."""
        if not chunks:
            return False
            
        sample_text = " ".join([c["text"] for c in chunks[:3]])
        
        # Heuristics:
        # 1. Emojis
        has_emojis = any(char in sample_text for char in ["😂", "😊", "👍", "❤️", "🤣"])
        # 2. Short, frequent turns (avg length < 60 chars)
        turns = sample_text.count("\n")
        avg_len = len(sample_text) / (turns + 1) if turns > 0 else 100
        
        return has_emojis or avg_len < 60

    async def extract_from_chunks(self, chunks: List[Dict[str, Any]], doc_type: str = "book") -> tuple[List[Entity], List[Relationship]]:
        """
        Extract entities and relationships from chunks using LangChain.
        """
        # Auto-detect social chat if it's a general conversation
        active_doc_type = doc_type
        if doc_type == "conversation" and self._is_social_chat(chunks):
            active_doc_type = "social_chat"
            print(f"[Graph] Auto-detected social_chat for {active_doc_type}")
            
        schema = DOC_TYPE_SCHEMAS.get(active_doc_type, DOC_TYPE_SCHEMAS["book"])
        batches = [chunks[i:i + self.batch_size] for i in range(0, len(chunks), self.batch_size)]

        tasks = [self._process_batch(batch, active_doc_type, schema) for batch in batches]
        results = await asyncio.gather(*tasks)

        all_entities = []
        all_relationships = []

        for res_entities, res_rels in results:
            all_entities.extend(res_entities)
            all_relationships.extend(res_rels)

        return all_entities, all_relationships

    async def _process_batch(self, batch: List[Dict[str, Any]], doc_type: str, schema: Dict[str, Any]):
        text_content = "\n\n".join([f"[{c['id']}]: {c['text']}" for c in batch])
        
        # Override prompt focus for social chats to be less formal
        focus_guidance = schema["focus"]
        if doc_type == "social_chat":
            focus_guidance += """
            CRITICAL: This is casual conversation. 
            - Focus on people mentioned (even non-speakers) and how they are connected or interacting.
            - Identify plans, shared interests, or mentioned events.
            - KINSHIP RESOLUTION: When you see relational references (e.g., "X's relative/friend"), ensure both parties are extracted as entities if possible.
              If the text reveals the identity of the relative, create entities for both individuals and link them with an appropriate relationship type.
            - Do NOT look for formal 'Decisions' or 'Action Items'.
            """
        
        max_retries = 5
        retry_delay = 5 

        # Small jitter
        await asyncio.sleep(random.uniform(0.05, 0.2))

        for attempt in range(max_retries):
            async with self.semaphore:
                try:
                    result: ExtractionResult = await self.entity_chain.ainvoke({
                        "doc_type": doc_type,
                        "entity_types": ", ".join(schema["entity_types"]),
                        "rel_types": ", ".join(schema["relationship_types"]),
                        "focus": focus_guidance,
                        "text_content": text_content
                    })

                    # Minimal cooldown
                    await asyncio.sleep(0.1)

                    # Convert to core domain models
                    entities = [
                        Entity(
                            name=e.name,
                            entity_type=e.entity_type,
                            description=e.description,
                            source_chunk_ids=e.source_chunk_ids
                        ) for e in result.entities
                    ]
                    
                    relationships = [
                        Relationship(
                            source_entity=r.source_entity,
                            target_entity=r.target_entity,
                            relation_type=r.relation_type,
                            description=r.description,
                            source_chunk_ids=r.source_chunk_ids
                        ) for r in result.relationships
                    ]
                    
                    return entities, relationships

                except Exception as e:
                    # Specific handling for Rate Limits
                    error_msg = str(e).lower()
                    if ("rate_limit" in error_msg or "429" in error_msg) and attempt < max_retries - 1:
                        wait_time = retry_delay * (2 ** attempt)
                        print(f"[Graph Extraction] Rate limit hit (Attempt {attempt+1}/{max_retries}). Retrying in {wait_time}s...")
                        await asyncio.sleep(wait_time)
                        continue
                    
                    print(f"[Graph Extraction Error] Batch failed: {e}")
                    return [], []
        
        return [], []

    async def extract_episodes(self, chunks: List[Dict[str, Any]]) -> List[Episode]:
        """Extract episodes from chunks."""
        # Skip episodes for social chats (too noisy/fluid)
        if self._is_social_chat(chunks):
            print("[Episode Extraction] Skipping for social chat")
            return []
            
        batches = [chunks[i:i + self.batch_size] for i in range(0, len(chunks), self.batch_size)]
        
        tasks = [self._process_episode_batch(batch) for batch in batches]
        results = await asyncio.gather(*tasks)
        
        all_episodes = []
        for batch_episodes in results:
            all_episodes.extend(batch_episodes)
            
        return all_episodes

    async def _process_episode_batch(self, batch: List[Dict[str, Any]]) -> List[Episode]:
        text_content = "\n\n".join([f"[{c['id']}]: {c['text']}" for c in batch])

        async with self.semaphore:
            try:
                result: EpisodeResult = await self.episode_chain.ainvoke({
                    "text_content": text_content
                })

                episodes = [
                    Episode(
                        speaker=ep.speaker,
                        topic=ep.topic,
                        stance=ep.stance,
                        summary=ep.summary,
                        entity_names=ep.mentioned_entities,
                        source_chunk_ids=ep.source_chunk_ids
                    ) for ep in result.episodes
                ]
                return episodes

            except Exception as e:
                print(f"[Episode Extraction Error]: {e}")
                return []
