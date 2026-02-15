from typing import List, Dict, Tuple
import re
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from src.graph.schemas import Entity, Relationship

class EntityResolver:
    """
    Resolves and deduplicates entities using semantic similarity.
    """
    
    def __init__(self, model_name: str = "BAAI/bge-small-en", threshold: float = 0.85):
        self.model = SentenceTransformer(model_name)
        self.threshold = threshold

    def resolve_kinship_references(self, entities: List[Entity], relationships: List[Relationship]) -> Tuple[List[Entity], List[Relationship]]:
        """
        Resolve possessive references like "X's sister" to actual people.
        Creates implied relationships if they don't exist.
        """
        kinship_pattern = re.compile(r"^(.+?)'s\s+(sister|brother|mother|father|mom|dad|friend|wife|husband|partner)$", re.I)

        new_relationships = list(relationships)
        
        # Helper to find entity by name (case-insensitive)
        def find_entity(name):
            return next((e for e in entities if e.name.lower() == name.lower()), None)

        for e in entities:
            match = kinship_pattern.match(e.name)
            if match:
                possessor_name = match.group(1)  # e.g., "Mia"
                relation = match.group(2)        # e.g., "sister"

                possessor = find_entity(possessor_name)

                if possessor:
                    # Create a relationship: "X's sister" connected_to "X"
                    # This ensures the graph is connected even if we don't know the sister's real name yet
                    # or links the "placeholder" entity to the anchor person.
                    
                    # Check if relationship already exists
                    exists = any(
                        r.source_entity == e.name and r.target_entity == possessor.name 
                        for r in new_relationships
                    )
                    
                    if not exists:
                        new_relationships.append(Relationship(
                            source_entity=e.name,
                            target_entity=possessor.name,
                            relation_type=f"{relation.lower()}_of",
                            description=f"Referenced as {possessor_name}'s {relation}",
                            source_chunk_ids=e.source_chunk_ids,
                            weight=0.8 # slightly lower confidence for implied
                        ))

        return entities, new_relationships

    def resolve(self, entities: List[Entity]) -> Tuple[List[Entity], Dict[str, str]]:
        """
        Deduplicate entities based on name similarity and description embedding.
        Returns (resolved_entities, name_mapping).
        name_mapping: {original_name: canonical_name}
        """
        if not entities:
            return [], {}

        # 1. Initial grouping by entity_type (only merge same types)
        by_type = {}
        for e in entities:
            if e.entity_type not in by_type:
                by_type[e.entity_type] = []
            by_type[e.entity_type].append(e)

        resolved_entities = []
        name_mapping = {}

        for e_type, group in by_type.items():
            if len(group) == 1:
                resolved_entities.extend(group)
                name_mapping[group[0].name] = group[0].name
                continue

            # 2. Embed descriptions (or names if desc is empty)
            texts = [
                f"{e.name}: {e.description}" if e.description else e.name 
                for e in group
            ]
            embeddings = self.model.encode(texts)

            # 3. Compute Similarity Matrix
            sim_matrix = cosine_similarity(embeddings)
            
            # 4. Connected Components (Union-Find)
            parent = list(range(len(group)))
            
            def find(i):
                if parent[i] != i:
                    parent[i] = find(parent[i])
                return parent[i]
            
            def union(i, j):
                root_i = find(i)
                root_j = find(j)
                if root_i != root_j:
                    parent[root_j] = root_i

            for i in range(len(group)):
                for j in range(i + 1, len(group)):
                    # Higher threshold (0.95) for person entities to avoid over-merging
                    current_threshold = 0.95 if e_type == "Person" else self.threshold
                    if sim_matrix[i][j] > current_threshold:
                        union(i, j)

            # 5. Merge Groups
            clusters = {}
            for i in range(len(group)):
                root = find(i)
                if root not in clusters:
                    clusters[root] = []
                clusters[root].append(group[i])

            for cluster in clusters.values():
                merged = self._merge_entities(cluster)
                resolved_entities.append(merged)
                for e in cluster:
                    name_mapping[e.name] = merged.name

        return resolved_entities, name_mapping

    def _merge_entities(self, cluster: List[Entity]) -> Entity:
        """Merge a list of entities into a single canonical entity."""
        if not cluster:
            raise ValueError("Empty cluster")
        if len(cluster) == 1:
            return cluster[0]

        # Canonical Name: Longest name (heuristic for specificity)
        # e.g. "Obama" vs "Barack Obama" -> "Barack Obama"
        canonical_name = max([e.name for e in cluster], key=len)
        
        # Description: Longest description
        descriptions = [e.description for e in cluster if e.description]
        canonical_desc = max(descriptions, key=len) if descriptions else None
        
        # Merge Source Chunk IDs
        all_chunk_ids = set()
        for e in cluster:
            all_chunk_ids.update(e.source_chunk_ids)
            
        # Merge Metadata
        merged_meta = {}
        for e in cluster:
            merged_meta.update(e.metadata)
            
        return Entity(
            name=canonical_name,
            entity_type=cluster[0].entity_type, # All same type in this cluster
            description=canonical_desc,
            metadata=merged_meta,
            source_chunk_ids=list(all_chunk_ids)
        )
