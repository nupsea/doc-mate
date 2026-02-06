import os
import sys
from src.graph.store import PostgresGraphStore

sys.path.append(os.getcwd())

def check_total_entities(doc_slug="ili"):
    store = PostgresGraphStore()
    doc_id = store._resolve_doc_id(doc_slug)
    
    if not doc_id:
        print(f"Document '{doc_slug}' not found.")
        return

    with store.conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM graph_entities WHERE doc_id = %s", (doc_id,))
        total = cur.fetchone()[0]
        print(f"Total entities for '{doc_slug}': {total}")
        
        if total > 0:
            print("First 20 entities:")
            cur.execute("SELECT name, entity_type FROM graph_entities WHERE doc_id = %s LIMIT 20", (doc_id,))
            for name, etype in cur.fetchall():
                print(f"  - {name} ({etype})")

if __name__ == "__main__":
    check_total_entities()
