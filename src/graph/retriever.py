from typing import List, Dict, Any
import logging
import re
from src.graph.store import PostgresGraphStore

logger = logging.getLogger(__name__)

class GraphRetriever:
    """
    Retrieves chunks based on graph traversal.
    """
    
    def __init__(self):
        self.store = PostgresGraphStore()

    def _extract_query_entities(self, query: str, doc_id: int, hint_entities: List[str] = None) -> List[int]:
        """
        Identify entities in the query by matching against known entities in the graph.
        Returns list of entity_ids.
        """
        # Fetch all entity names for this doc
        with self.store.conn.cursor() as cur:
            cur.execute("SELECT name, entity_id FROM graph_entities WHERE doc_id = %s", (doc_id,))
            rows = cur.fetchall()
            
        found_ids = []
        query_lower = query.lower()
        
        # Tokenize query to find individual words
        query_words = set(re.findall(r'\w+', query_lower))
        
        # Also include hint entities from router
        if hint_entities:
            query_words.update(w.lower() for w in hint_entities)
        
        for name, eid in rows:
            name_lower = name.lower()
            name_words = set(re.findall(r'\w+', name_lower))
            
            # Match if:
            # 1. Full name is in query
            if name_lower in query_lower:
                found_ids.append(eid)
                continue
                
            # 2. Any significant word from the query matches the entity name
            matching_words = query_words.intersection(name_words)
            significant_matches = [w for w in matching_words if len(w) > 3]
            
            if significant_matches:
                found_ids.append(eid)
                
        return found_ids

    def search(self, query: str, doc_slug: str, topk: int = 7, hint_entities: List[str] = None) -> List[Dict[str, Any]]:
        """
        Search for chunks related to entities in the query.
        Returns list of dicts with 'id' and 'score'.
        """
        doc_id = self.store._resolve_doc_id(doc_slug)
        if not doc_id:
            logger.warning(f"Graph search failed: doc '{doc_slug}' not found")
            return []

        # 1. Identify starting points (including hint entities)
        entity_ids = self._extract_query_entities(query, doc_id, hint_entities)
        
        # 2. Relationship type matching (Priority 2 fix)
        # If query contains "brother", "sister", etc., find relationships of that type
        kinship_terms = {
            "brother": ["sibling_of", "brother_of", "related_to", "connected_to"],
            "sister": ["sibling_of", "sister_of", "related_to", "connected_to"],
            "mother": ["parent_of", "mother_of", "related_to", "connected_to"],
            "mom": ["parent_of", "mother_of", "related_to", "connected_to"],
            "father": ["parent_of", "father_of", "related_to", "connected_to"],
            "dad": ["parent_of", "father_of", "related_to", "connected_to"],
            "friend": ["friend_of", "connected_to", "interacts_with"],
            "dating": ["dates", "interacts_with", "connected_to"],
            "partner": ["dates", "interacts_with", "connected_to", "connected_to"],
        }

        query_lower = query.lower()
        for term, rel_types in kinship_terms.items():
            if term in query_lower:
                with self.store.conn.cursor() as cur:
                    cur.execute(
                        """
                        SELECT DISTINCT source_entity_id, target_entity_id
                        FROM graph_relationships
                        WHERE doc_id = %s AND relation_type = ANY(%s)
                        """,
                        (doc_id, rel_types)
                    )
                    for row in cur.fetchall():
                        if row[0] not in entity_ids: 
                            entity_ids.append(row[0])
                        if row[1] not in entity_ids:
                            entity_ids.append(row[1])

        if not entity_ids:
            return []

        # 3. Traverse Graph (2 hops)
        # Collect chunk IDs with scoring based on distance
        chunk_scores = {}  # {chunk_id: score}
        
        for eid in entity_ids:
            # Direct mention (Hop 0) -> Score 1.0
            direct_chunks = self.store.get_chunk_ids_for_entities([eid])
            for cid in direct_chunks:
                chunk_scores[cid] = chunk_scores.get(cid, 0) + 1.0
            
            # Related entities (Hop 1 & 2)
            related = self.store.find_related_entities(eid, hops=2)
            
            for rel in related:
                dist = rel['depth']
                weight = 0.5 if dist == 1 else 0.25
                
                for cid in rel['source_chunk_ids']:
                    chunk_scores[cid] = chunk_scores.get(cid, 0) + weight

        # 4. Format Results
        results = [
            {"id": cid, "score": score}
            for cid, score in chunk_scores.items()
        ]
        
        # Sort by score descending
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:topk]

class ConversationGraphRetriever(GraphRetriever):
    """
    Specialized retriever for conversation/transcript graphs.
    Includes episode matching.
    """
    
    def search(self, query: str, doc_slug: str, topk: int = 7, hint_entities: List[str] = None) -> List[Dict[str, Any]]:
        # 1. Standard Entity Search
        base_results = super().search(query, doc_slug, topk, hint_entities)
        
        doc_id = self.store._resolve_doc_id(doc_slug)
        if not doc_id:
            return base_results

        # 2. Episode Search
        # Match query text against episode topics, speakers, or summaries
        # This is a bit brute-force SQL ILIKE, but efficient for reasonable doc sizes
        with self.store.conn.cursor() as cur:
            # Simple keyword matching for speaker or topic
            # e.g. "What did Alice say about the budget?"
            cur.execute(
                """
                SELECT source_chunk_ids, speaker, topic 
                FROM graph_episodes 
                WHERE doc_id = %s 
                AND (
                    %s ILIKE '%%' || speaker || '%%' 
                    OR %s ILIKE '%%' || topic || '%%'
                )
                """,
                (doc_id, query, query)
            )
            rows = cur.fetchall()
            
        episode_chunk_scores = {}
        for row in rows:
            chunk_ids = row[0]
            for cid in chunk_ids:
                episode_chunk_scores[cid] = episode_chunk_scores.get(cid, 0) + 2.0  # High weight for episode match
                
        # 3. Merge Results
        final_scores = {r["id"]: r["score"] for r in base_results}
        
        for cid, score in episode_chunk_scores.items():
            final_scores[cid] = final_scores.get(cid, 0) + score
            
        final_results = [
            {"id": cid, "score": score}
            for cid, score in final_scores.items()
        ]
        final_results.sort(key=lambda x: x["score"], reverse=True)
        
        return final_results[:topk]
