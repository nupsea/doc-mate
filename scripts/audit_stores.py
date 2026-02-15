"""
Audit script: per-document consistency check across Postgres (BM25, graph)
and Qdrant (vectors).

Reports chunk counts per slug for each store and flags mismatches.

Usage:
    uv run python scripts/audit_stores.py
"""

import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.content.store import PgresStore


def get_documents(store):
    """Fetch all documents with their expected chunk counts."""
    rows = store.execute(
        "SELECT doc_id, slug, title, num_chunks, doc_type FROM documents ORDER BY slug",
        fetch="all",
    )
    return [
        {"doc_id": r[0], "slug": r[1], "title": r[2], "num_chunks": r[3] or 0, "doc_type": r[4]}
        for r in rows
    ] if rows else []


def get_bm25_counts(store):
    """Chunk counts per doc_id in bm25_doc_lens."""
    rows = store.execute(
        "SELECT doc_id, COUNT(*) FROM bm25_doc_lens GROUP BY doc_id",
        fetch="all",
    )
    return {r[0]: r[1] for r in rows} if rows else {}


def get_bm25_term_counts(store):
    """Total term-frequency rows per doc_id in bm25_index."""
    rows = store.execute(
        "SELECT doc_id, COUNT(*) FROM bm25_index GROUP BY doc_id",
        fetch="all",
    )
    return {r[0]: r[1] for r in rows} if rows else {}


def get_graph_counts(store):
    """Entity and relationship counts per doc_id."""
    entity_rows = store.execute(
        "SELECT doc_id, COUNT(*) FROM graph_entities GROUP BY doc_id",
        fetch="all",
    )
    rel_rows = store.execute(
        "SELECT doc_id, COUNT(*) FROM graph_relationships GROUP BY doc_id",
        fetch="all",
    )
    episode_rows = store.execute(
        "SELECT doc_id, COUNT(*) FROM graph_episodes GROUP BY doc_id",
        fetch="all",
    )
    entities = {r[0]: r[1] for r in entity_rows} if entity_rows else {}
    rels = {r[0]: r[1] for r in rel_rows} if rel_rows else {}
    episodes = {r[0]: r[1] for r in episode_rows} if episode_rows else {}
    return entities, rels, episodes


def get_qdrant_counts():
    """Chunk counts per slug prefix in Qdrant."""
    try:
        from src.search.vec import SemanticRetriever
        vec = SemanticRetriever()
        if not vec.qdrant.collection_exists(vec.COLLECTION):
            return {}, 0

        counts = {}
        total = 0
        for chunk in vec.get_all_chunks():
            chunk_id = chunk.get("id", "")
            parts = chunk_id.split("_")
            slug_parts = []
            for p in parts:
                if p.isdigit():
                    break
                slug_parts.append(p)
            slug = "_".join(slug_parts)
            counts[slug] = counts.get(slug, 0) + 1
            total += 1
        return counts, total
    except Exception as e:
        print(f"  [WARNING] Qdrant unavailable: {e}")
        return {}, 0


def get_summary_counts(store):
    """Chapter summary and document summary counts per doc_id."""
    ch_rows = store.execute(
        "SELECT doc_id, COUNT(*) FROM chapter_summaries GROUP BY doc_id",
        fetch="all",
    )
    doc_rows = store.execute(
        "SELECT doc_id FROM document_summaries",
        fetch="all",
    )
    chapters = {r[0]: r[1] for r in ch_rows} if ch_rows else {}
    has_summary = {r[0] for r in doc_rows} if doc_rows else set()
    return chapters, has_summary


def main():
    store = PgresStore()
    docs = get_documents(store)

    if not docs:
        print("No documents found.")
        return

    # Gather all stats
    bm25_chunks = get_bm25_counts(store)
    bm25_terms = get_bm25_term_counts(store)
    graph_entities, graph_rels, graph_episodes = get_graph_counts(store)
    chapter_counts, has_doc_summary = get_summary_counts(store)
    qdrant_counts, qdrant_total = get_qdrant_counts()

    # Check for orphaned Qdrant slugs
    valid_slugs = {d["slug"] for d in docs}
    orphaned_qdrant = {s: c for s, c in qdrant_counts.items() if s not in valid_slugs}

    # Header
    print(f"{'Slug':<20} {'Type':<12} {'Expected':>8} {'BM25':>8} {'Vector':>8} {'Terms':>8} {'Entity':>8} {'Rels':>8} {'Epis':>8} {'ChSum':>8} {'DocSum':>8} {'Issues'}")
    print("-" * 140)

    issues_found = 0

    for doc in docs:
        did = doc["doc_id"]
        slug = doc["slug"]
        expected = doc["num_chunks"]
        dtype = doc["doc_type"]

        bm25 = bm25_chunks.get(did, 0)
        vec = qdrant_counts.get(slug, 0)
        terms = bm25_terms.get(did, 0)
        ent = graph_entities.get(did, 0)
        rels = graph_rels.get(did, 0)
        epis = graph_episodes.get(did, 0)
        ch_sum = chapter_counts.get(did, 0)
        d_sum = "Y" if did in has_doc_summary else "-"

        # Flag mismatches
        flags = []
        if bm25 != expected:
            flags.append(f"bm25({bm25})!=expected({expected})")
        if vec != expected:
            flags.append(f"vec({vec})!=expected({expected})")
        if bm25 != vec:
            flags.append(f"bm25({bm25})!=vec({vec})")
        if ent == 0 and expected > 0:
            flags.append("no_graph")

        flag_str = ", ".join(flags) if flags else ""
        if flags:
            issues_found += 1

        print(f"{slug:<20} {dtype:<12} {expected:>8} {bm25:>8} {vec:>8} {terms:>8} {ent:>8} {rels:>8} {epis:>8} {ch_sum:>8} {d_sum:>8} {flag_str}")

    print("-" * 140)

    # Totals
    total_expected = sum(d["num_chunks"] for d in docs)
    total_bm25 = sum(bm25_chunks.values())
    total_terms = sum(bm25_terms.values())
    total_ent = sum(graph_entities.values())
    total_rels = sum(graph_rels.values())
    total_epis = sum(graph_episodes.values())
    total_ch = sum(chapter_counts.values())
    total_dsum = len(has_doc_summary)

    print(f"{'TOTAL':<20} {'':<12} {total_expected:>8} {total_bm25:>8} {qdrant_total:>8} {total_terms:>8} {total_ent:>8} {total_rels:>8} {total_epis:>8} {total_ch:>8} {total_dsum:>8}")
    print()

    # Orphaned Qdrant vectors
    if orphaned_qdrant:
        print("ORPHANED QDRANT VECTORS (no matching document):")
        for slug, count in sorted(orphaned_qdrant.items()):
            print(f"  {slug}: {count} chunks")
        print()

    if issues_found:
        print(f"{issues_found} document(s) with mismatches.")
    else:
        print("All stores consistent.")


if __name__ == "__main__":
    main()
