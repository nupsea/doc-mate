from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field

class Entity(BaseModel):
    """Represents a named entity extracted from text."""
    name: str
    entity_type: str
    description: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    source_chunk_ids: List[str] = Field(default_factory=list)
    entity_id: Optional[int] = None  # DB ID

class Relationship(BaseModel):
    """Represents a relationship between two entities."""
    source_entity: str
    target_entity: str
    relation_type: str
    weight: float = 1.0
    description: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    source_chunk_ids: List[str] = Field(default_factory=list)
    rel_id: Optional[int] = None  # DB ID

class Episode(BaseModel):
    """Represents a distinct episode or conversation turn."""
    speaker: str
    stance: Optional[str] = None
    topic: Optional[str] = None
    summary: Optional[str] = None
    turn_start: Optional[int] = None
    turn_end: Optional[int] = None
    timestamp_start: Optional[str] = None
    timestamp_end: Optional[str] = None
    entity_names: List[str] = Field(default_factory=list)
    source_chunk_ids: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    episode_id: Optional[int] = None  # DB ID

# Schema definitions for different document types to guide extraction
DOC_TYPE_SCHEMAS = {
    "book": {
        "entity_types": ["Person", "Location", "Organization", "Event", "Concept"],
        "relationship_types": ["knows", "located_at", "involved_in", "causes", "member_of"],
        "focus": "Character relationships, plot events, and thematic concepts."
    },
    "script": {
        "entity_types": ["Character", "Scene", "Prop", "Action"],
        "relationship_types": ["appears_in", "uses", "interacts_with"],
        "focus": "Character interactions per scene."
    },
    "conversation": {
        "entity_types": ["Speaker", "Topic", "Decision", "ActionItem"],
        "relationship_types": ["advocates", "opposes", "agrees_with", "assigned_to"],
        "focus": "Speaker stances, decisions made, and action items."
    },
    "social_chat": {
        "entity_types": ["Person", "Location", "Event", "SharedContent", "Topic"],
        "relationship_types": ["mentions", "interacts_with", "plans_with", "connected_to", "discusses"],
        "focus": "Interpersonal dynamics, social connections, shared events, and plans between people."
    },
    "tech_doc": {
        "entity_types": ["Component", "Function", "Class", "API", "Concept"],
        "relationship_types": ["calls", "inherits_from", "depends_on", "implements"],
        "focus": "System architecture, API dependencies, and data flow."
    },
    "report": {
        "entity_types": ["Metric", "Company", "Trend", "Risk", "Strategy"],
        "relationship_types": ["increases", "decreases", "impacts", "correlated_with"],
        "focus": "Key performance indicators, trends, and causal factors."
    }
}
