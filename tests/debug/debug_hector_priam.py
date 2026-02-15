import os
import sys
from src.graph.store import PostgresGraphStore

sys.path.append(os.getcwd())

def inspect_hector_priam(doc_slug="ili"):
    store = PostgresGraphStore()
    doc_id = store._resolve_doc_id(doc_slug)
    
    if not doc_id:
        print(f"Document '{doc_slug}' not found.")
        return

    print(f"Inspecting Hector & Priam in '{doc_slug}' (ID: {doc_id}):")
    
    entities = ["%Hector%", "%Priam%"]
    found_entities = {}
    
    with store.conn.cursor() as cur:
        for ent_pattern in entities:
            cur.execute("SELECT entity_id, name, entity_type FROM graph_entities WHERE doc_id = %s AND name ILIKE %s", (doc_id, ent_pattern))
            rows = cur.fetchall()
            print(f"- Found for {ent_pattern}: {rows}")
            for eid, name, etype in rows:
                found_entities[name] = eid
        
        # Check relationships between them if both found
        if len(found_entities) >= 2:
            names = list(found_entities.keys())
            for i in range(len(names)):
                for j in range(i + 1, len(names)):
                    name1, name2 = names[i], names[j]
                    id1, id2 = found_entities[name1], found_entities[name2]
                    
                    cur.execute("""
                        SELECT relation_type, description 
                        FROM graph_relationships 
                        WHERE doc_id = %s 
                        AND (
                            (source_entity_id = %s AND target_entity_id = %s)
                            OR (source_entity_id = %s AND target_entity_id = %s)
                        )
                    """, (doc_id, id1, id2, id2, id1))
                    
                    rels = cur.fetchall()
                    if rels:
                        print(f"- Relationship between {name1} and {name2}:")
                        for rtype, desc in rels:
                            print(f"  -> {rtype}: {desc}")
                    else:
                        print(f"- NO DIRECT relationship found between {name1} and {name2}")
        
        # Check 1-hop for each
        for name, eid in found_entities.items():
            print(f"- Top 5 relationships for {name}:")
            rels = store.find_related_entities(eid, hops=1)
            for r in rels[:5]:
                print(f"  -> {r['relation_type']} {r['name']} ({r['entity_type']})")

if __name__ == "__main__":
    inspect_hector_priam()
