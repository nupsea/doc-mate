import os
import sys
from src.graph.store import PostgresGraphStore

sys.path.append(os.getcwd())

def inspect_graph(doc_slug="ili"):
    store = PostgresGraphStore()
    doc_id = store._resolve_doc_id(doc_slug)
    
    if not doc_id:
        print(f"Document '{doc_slug}' not found.")
        return

    print(f"Graph content for '{doc_slug}' (ID: {doc_id}):")
    
    # Check entities
    with store.conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM graph_entities WHERE doc_id = %s", (doc_id,))
        count = cur.fetchone()[0]
        print(f"- Entities: {count}")
        
        # Check specific entities
        # Use simple string concatenation for the pattern to avoid param/wildcard confusion
        term = '%Achilles%'
        cur.execute("SELECT entity_id, name, entity_type FROM graph_entities WHERE doc_id = %s AND name ILIKE %s", (doc_id, term))
        rows = cur.fetchall()
        print(f"- Found 'Achilles' entities: {rows}")
        
        if rows:
            achilles_id = rows[0][0]
            # Check relationships
            print(f"- Relationships for Achilles (ID {achilles_id}):")
            rels = store.find_related_entities(achilles_id, hops=1)
            for r in rels:
                print(f"  -> {r['relation_type']} {r['name']} ({r['entity_type']})")
        else:
            print("- Achilles not found in graph!")

if __name__ == "__main__":
    inspect_graph()
