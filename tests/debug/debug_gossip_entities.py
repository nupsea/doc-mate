from src.graph.store import PostgresGraphStore

def inspect_gossip(doc_slug="gossip_chat"):
    store = PostgresGraphStore()
    doc_id = store._resolve_doc_id(doc_slug)
    if not doc_id:
        print(f"Document {doc_slug} not found.")
        return

    with store.conn.cursor() as cur:
        # All entities
        cur.execute("SELECT name, entity_type, description FROM graph_entities WHERE doc_id = %s", (doc_id,))
        print("=== ENTITIES ===")
        for name, etype, desc in cur.fetchall():
            print(f"  [{etype}] {name}: {desc[:50] if desc else 'N/A'}...")

        # All relationships
        cur.execute("""
            SELECT e1.name, r.relation_type, e2.name
            FROM graph_relationships r
            JOIN graph_entities e1 ON r.source_entity_id = e1.entity_id
            JOIN graph_entities e2 ON r.target_entity_id = e2.entity_id
            WHERE r.doc_id = %s
        """, (doc_id,))
        print("\n=== RELATIONSHIPS ===")
        for src, rel, tgt in cur.fetchall():
            print(f"  {src} --[{rel}]--> {tgt}")

if __name__ == "__main__":
    inspect_gossip()