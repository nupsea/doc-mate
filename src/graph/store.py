from typing import List, Dict, Any
import json
from src.content.store import PgresStore
from src.graph.schemas import Entity, Relationship, Episode
from psycopg2.extras import execute_values

class PostgresGraphStore(PgresStore):
    """
    Postgres-based store for Graph Knowledge Layer.
    Handles storage and retrieval of entities, relationships, and episodes.
    """

    def store_entities(self, doc_id: int, entities: List[Entity]) -> List[int]:
        """
        Store extracted entities. Updates if entity (doc_id, name, type) exists.
        Returns list of entity_ids.
        """
        if not entities:
            return []

        # Deduplicate entities in Python first to avoid "ON CONFLICT ... cannot affect row a second time"
        # Merge logic: Combine chunk_ids, overwrite description if new one is longer, merge metadata
        unique_entities = {}
        for e in entities:
            # Normalize: strip whitespace and handle case
            norm_name = e.name.strip()
            if not norm_name:
                continue
                
            key = (norm_name, e.entity_type)
            if key not in unique_entities:
                unique_entities[key] = e
                e.name = norm_name # Use normalized name
            else:
                existing = unique_entities[key]
                # Merge chunk IDs
                existing.source_chunk_ids = list(set(existing.source_chunk_ids + e.source_chunk_ids))
                # Merge metadata
                existing.metadata.update(e.metadata)
                # Keep longer description
                if e.description and len(e.description) > len(existing.description or ""):
                    existing.description = e.description

        rows = [
            (
                doc_id,
                e.name,
                e.entity_type,
                e.description,
                json.dumps(e.metadata),
                e.source_chunk_ids
            )
            for e in unique_entities.values()
        ]

        with self._get_conn() as conn:
            with conn.cursor() as cur:
                execute_values(
                    cur,
                    """
                    INSERT INTO graph_entities (doc_id, name, entity_type, description, metadata, source_chunk_ids)
                    VALUES %s
                    ON CONFLICT (doc_id, name, entity_type) 
                    DO UPDATE SET
                        description = COALESCE(excluded.description, graph_entities.description),
                        metadata = graph_entities.metadata || excluded.metadata,
                        source_chunk_ids = (
                            SELECT array_agg(DISTINCT x) 
                            FROM unnest(graph_entities.source_chunk_ids || excluded.source_chunk_ids) AS x
                        )
                    """,
                    rows
                )
            conn.commit()

        # After storing, fetch ALL entity IDs for this document to ensure stats are accurate
        # and IDs are available for relationship resolution.
        all_names = [e.name for e in unique_entities.values()]
        name_to_id = self.find_entities_by_names(doc_id, all_names)
        
        return list(name_to_id.values())

    def store_relationships(self, doc_id: int, relationships: List[Relationship]) -> int:
        """
        Store relationships between entities.
        Resolves entity names to IDs before insertion.
        Deduplicates and accumulates weights on conflict.
        Returns count of inserted relationships.
        """
        if not relationships:
            return 0

        # First, resolve all entity names to IDs for this doc
        entity_names = set()
        for r in relationships:
            entity_names.add(r.source_entity)
            entity_names.add(r.target_entity)
        
        name_to_id = self.find_entities_by_names(doc_id, list(entity_names))
        
        # Deduplicate in Python to avoid "ON CONFLICT ... cannot affect row a second time"
        unique_rels = {}
        for r in relationships:
            source_id = name_to_id.get(r.source_entity)
            target_id = name_to_id.get(r.target_entity)
            
            # Use original name if direct ID lookup fails (maybe case difference in extractor)
            if not source_id:
                # Try case-insensitive lookup
                for name, eid in name_to_id.items():
                    if name.lower() == r.source_entity.lower():
                        source_id = eid
                        break
            if not target_id:
                for name, eid in name_to_id.items():
                    if name.lower() == r.target_entity.lower():
                        target_id = eid
                        break

            if source_id and target_id:
                key = (source_id, target_id, r.relation_type)
                if key not in unique_rels:
                    unique_rels[key] = {
                        "doc_id": doc_id,
                        "source_id": source_id,
                        "target_id": target_id,
                        "type": r.relation_type,
                        "weight": r.weight,
                        "desc": r.description,
                        "meta": r.metadata,
                        "chunks": r.source_chunk_ids
                    }
                else:
                    existing = unique_rels[key]
                    existing["weight"] += r.weight
                    existing["chunks"] = list(set(existing["chunks"] + r.source_chunk_ids))
                    if r.description and len(r.description) > len(existing["desc"] or ""):
                        existing["desc"] = r.description

        rows = [
            (
                v["doc_id"],
                v["source_id"],
                v["target_id"],
                v["type"],
                v["weight"],
                v["desc"],
                json.dumps(v["meta"]),
                v["chunks"]
            )
            for v in unique_rels.values()
        ]
        
        if not rows:
            return 0

        with self._get_conn() as conn:
            with conn.cursor() as cur:
                execute_values(
                    cur,
                    """
                    INSERT INTO graph_relationships 
                    (doc_id, source_entity_id, target_entity_id, relation_type, weight, description, metadata, source_chunk_ids)
                    VALUES %s
                    ON CONFLICT (doc_id, source_entity_id, target_entity_id, relation_type) 
                    DO UPDATE SET
                        weight = graph_relationships.weight + EXCLUDED.weight,
                        description = COALESCE(graph_relationships.description, EXCLUDED.description),
                        source_chunk_ids = (
                            SELECT array_agg(DISTINCT x) 
                            FROM unnest(graph_relationships.source_chunk_ids || EXCLUDED.source_chunk_ids) AS x
                        )
                    """,
                    rows
                )
                count = cur.rowcount
            conn.commit()
        return count

    def store_episodes(self, doc_id: int, episodes: List[Episode]) -> int:
        """
        Store conversation episodes.
        Returns count of inserted episodes.
        """
        if not episodes:
            return 0

        rows = [
            (
                doc_id,
                ep.speaker,
                ep.stance,
                ep.topic,
                ep.summary,
                ep.turn_start,
                ep.turn_end,
                ep.timestamp_start,
                ep.timestamp_end,
                # Resolve entity names to IDs if possible, else store empty array for now
                # In a full impl, we'd resolve these like relationships
                [], 
                ep.source_chunk_ids,
                json.dumps(ep.metadata)
            )
            for ep in episodes
        ]

        with self._get_conn() as conn:
            with conn.cursor() as cur:
                execute_values(
                    cur,
                    """
                    INSERT INTO graph_episodes 
                    (doc_id, speaker, stance, topic, summary, turn_start, turn_end, timestamp_start, timestamp_end, entity_ids, source_chunk_ids, metadata)
                    VALUES %s
                    """,
                    rows
                )
                count = cur.rowcount
            conn.commit()
        return count

    def find_entities_by_names(self, doc_id: int, names: List[str]) -> Dict[str, int]:
        """
        Look up entity IDs by name (case-insensitive match).
        Returns dict {name: entity_id}.
        """
        if not names:
            return {}
            
        # Normalize names to lower for search
        search_names = [n.lower() for n in names]
        rows = self.execute(
            "SELECT name, entity_id FROM graph_entities WHERE doc_id = %s AND LOWER(name) = ANY(%s)",
            (doc_id, search_names),
            fetch="all"
        )
        return {row[0]: row[1] for row in rows} if rows else {}

    def find_docs_by_entities(self, names: List[str]) -> List[str]:
        """
        Find document slugs that contain the given entities.
        Useful for routing queries to documents based on content.
        """
        if not names:
            return []
            
        search_names = [n.lower() for n in names]
        rows = self.execute(
            """
            SELECT DISTINCT d.slug
            FROM graph_entities ge
            JOIN documents d ON ge.doc_id = d.doc_id
            WHERE LOWER(ge.name) = ANY(%s)
            """,
            (search_names,),
            fetch="all"
        )
        return [row[0] for row in rows] if rows else []

    def find_related_entities(self, entity_id: int, hops: int = 2) -> List[Dict[str, Any]]:
        """
        Find entities related to the given entity_id within 'hops' distance.
        Uses recursive CTE for graph traversal.
        """
        rows = self.execute(
            """
            WITH RECURSIVE graph_walk AS (
                -- Base case: Direct neighbors (Hop 1)
                SELECT 
                    CASE 
                        WHEN r.source_entity_id = %s THEN r.target_entity_id
                        ELSE r.source_entity_id 
                    END AS entity_id,
                    r.relation_type,
                    1 AS depth
                FROM graph_relationships r
                WHERE r.source_entity_id = %s OR r.target_entity_id = %s

                UNION ALL

                -- Recursive step: Neighbors of neighbors (Hop N)
                SELECT 
                    CASE 
                        WHEN r2.source_entity_id = gw.entity_id THEN r2.target_entity_id
                        ELSE r2.source_entity_id 
                    END AS entity_id,
                    r2.relation_type,
                    gw.depth + 1
                FROM graph_walk gw
                JOIN graph_relationships r2 ON r2.source_entity_id = gw.entity_id OR r2.target_entity_id = gw.entity_id
                WHERE gw.depth < %s
            )
            SELECT DISTINCT 
                e.entity_id, 
                e.name, 
                e.entity_type, 
                e.description,
                gw.relation_type, 
                gw.depth,
                e.source_chunk_ids
            FROM graph_walk gw
            JOIN graph_entities e ON e.entity_id = gw.entity_id
            WHERE e.entity_id != %s
            ORDER BY gw.depth, e.name
            """,
            (entity_id, entity_id, entity_id, hops, entity_id),
            fetch="all"
        )
        
        if not rows:
            return []
            
        results = []
        for row in rows:
            results.append({
                "entity_id": row[0],
                "name": row[1],
                "entity_type": row[2],
                "description": row[3],
                "relation_type": row[4],
                "depth": row[5],
                "source_chunk_ids": row[6]
            })
        return results

    def get_chunk_ids_for_entities(self, entity_ids: List[int]) -> List[str]:
        """Collect all source_chunk_ids associated with a list of entities."""
        if not entity_ids:
            return []
            
        rows = self.execute(
            "SELECT DISTINCT unnest(source_chunk_ids) FROM graph_entities WHERE entity_id = ANY(%s)",
            (entity_ids,),
            fetch="all"
        )
        return [row[0] for row in rows] if rows else []

    def delete_graph_for_doc(self, doc_id: int):
        """
        Delete all graph data for a document.
        (Note: Cascade delete on documents table usually handles this, 
        but this is useful for re-ingestion).
        """
        self.execute("DELETE FROM graph_entities WHERE doc_id = %s", (doc_id,), commit=True)
        self.execute("DELETE FROM graph_episodes WHERE doc_id = %s", (doc_id,), commit=True)

    def get_episodes_for_doc(self, slug: str, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Fetch recent episodes for a document (conversation).
        Useful for providing high-level conversation flow context.
        """
        doc_id = self._resolve_doc_id(slug)
        if not doc_id:
            return []

        rows = self.execute(
            """
            SELECT topic, summary, speaker, stance 
            FROM graph_episodes 
            WHERE doc_id = %s 
            ORDER BY turn_start ASC 
            LIMIT %s
            """,
            (doc_id, limit),
            fetch="all"
        )
        
        if not rows:
            return []
            
        episodes = []
        for row in rows:
            episodes.append({
                "topic": row[0],
                "summary": row[1],
                "speaker": row[2],
                "stance": row[3]
            })
        return episodes
