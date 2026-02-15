"""
Cleanup script: find and remove orphaned rows in bm25_index and bm25_doc_lens
whose chunk_id prefix (slug) does not match any slug in the documents table.

Usage:
    uv run python scripts/cleanup_orphaned_bm25.py          # dry-run (report only)
    uv run python scripts/cleanup_orphaned_bm25.py --apply   # actually delete
"""

import argparse
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.content.store import PgresStore

TABLES = ["bm25_index", "bm25_doc_lens"]


def get_valid_slugs(store: PgresStore) -> set[str]:
    rows = store.execute("SELECT slug FROM documents", fetch="all")
    return {r[0] for r in rows} if rows else set()


def count_orphaned(store: PgresStore, table: str) -> dict[str, int]:
    """
    Find chunk_ids in `table` that don't match any document slug prefix.
    Returns {orphaned_slug_guess: row_count}.
    Uses a NOT EXISTS subquery against documents for correctness.
    """
    # Get distinct orphaned chunk_ids grouped by their likely slug prefix
    rows = store.execute(
        f"""
        SELECT substring(t.chunk_id FROM '^[a-z_]+') AS prefix, COUNT(*) AS cnt
        FROM {table} t
        WHERE NOT EXISTS (
            SELECT 1 FROM documents d WHERE t.chunk_id LIKE d.slug || '_%'
        )
        GROUP BY prefix
        ORDER BY prefix
        """,
        fetch="all",
    )
    return {r[0]: r[1] for r in rows} if rows else {}


def delete_all_orphaned(store: PgresStore, table: str) -> int:
    """Delete all orphaned rows from `table`. Returns count deleted."""
    store.execute(
        f"""
        DELETE FROM {table} t
        WHERE NOT EXISTS (
            SELECT 1 FROM documents d WHERE t.chunk_id LIKE d.slug || '_%'
        )
        """,
        commit=True,
    )
    # Verify none remain
    row = store.execute(
        f"""
        SELECT COUNT(*) FROM {table} t
        WHERE NOT EXISTS (
            SELECT 1 FROM documents d WHERE t.chunk_id LIKE d.slug || '_%'
        )
        """,
        fetch="one",
    )
    return row[0] if row else 0


def main():
    parser = argparse.ArgumentParser(description="Clean up orphaned BM25 rows")
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually delete orphaned rows (default is dry-run)",
    )
    args = parser.parse_args()

    store = PgresStore()
    valid_slugs = get_valid_slugs(store)
    print(f"Documents table has {len(valid_slugs)} slug(s): {sorted(valid_slugs)}\n")

    found_orphans = False

    for table in TABLES:
        total_row = store.execute(f"SELECT COUNT(*) FROM {table}", fetch="one")
        total = total_row[0] if total_row else 0
        print(f"--- {table} ({total} total rows) ---")

        orphans = count_orphaned(store, table)
        if not orphans:
            print("  No orphaned rows found.\n")
            continue

        found_orphans = True
        for prefix, count in orphans.items():
            print(f"  Orphaned prefix '{prefix}': {count} rows")

        if args.apply:
            remaining = delete_all_orphaned(store, table)
            print(f"  Deleted. Remaining orphans: {remaining}\n")
        else:
            print("  (dry-run -- use --apply to delete)\n")

    if not found_orphans:
        print("Nothing to clean up.")
    elif not args.apply:
        print("Re-run with --apply to delete the orphaned rows.")


if __name__ == "__main__":
    main()
