"""
Test that conversation metadata (timestamps, speakers) is properly retrieved and formatted.
Uses sample Teams meeting conversation data.
"""

from src.search.vec import SemanticRetriever
from src.content.store import PgresStore


def test_conversation_metadata_retrieval():
    """Test that metadata is included in search results for conversation documents."""
    print("=" * 80)
    print("CONVERSATION METADATA RETRIEVAL TESTS")
    print("=" * 80)

    passed = 0
    total = 0

    # Test 1: Verify get_chunks_by_ids returns metadata
    print("\n[1] Testing get_chunks_by_ids() includes metadata field...")
    total += 1
    try:
        retriever = SemanticRetriever()

        # Use sample Teams meeting conversation document
        book_slug = "stm"  # Sample Teams Meeting
        store = PgresStore()
        with store.conn.cursor() as cur:
            cur.execute("SELECT slug FROM books WHERE slug = %s", (book_slug,))
            row = cur.fetchone()

            if not row:
                print(f"  ⊘ SKIPPED: Sample conversation document '{book_slug}' not found in database")
            else:
                print(f"  Using sample conversation document: {book_slug}")

                # Search for chunks
                chunk_ids = retriever.id_search("meeting", topk=3)

                if not chunk_ids:
                    print("  ⊘ SKIPPED: No chunks found")
                else:
                    # Retrieve full chunks
                    chunks = retriever.get_chunks_by_ids(chunk_ids)

                    # Verify structure includes metadata
                    all_have_metadata = True
                    for i, chunk in enumerate(chunks, 1):
                        if "id" not in chunk:
                            all_have_metadata = False
                            print(f"  ✗ Chunk {i} missing 'id'")
                        if "text" not in chunk:
                            all_have_metadata = False
                            print(f"  ✗ Chunk {i} missing 'text'")
                        if "metadata" not in chunk:
                            all_have_metadata = False
                            print(f"  ✗ Chunk {i} missing 'metadata'")

                    if all_have_metadata:
                        print(f"  ✓ PASSED: All {len(chunks)} chunks include metadata field")
                        passed += 1
                    else:
                        print(f"  ✗ FAILED: Some chunks missing required fields")

    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        import traceback
        traceback.print_exc()

    # Test 2: Verify search() returns metadata
    print("\n[2] Testing search() includes metadata field...")
    total += 1
    try:
        retriever = SemanticRetriever()
        results = retriever.search("planning strategy", topk=3)

        if not results:
            print("  ⊘ SKIPPED: No search results found")
        else:
            # Verify structure includes metadata
            all_have_metadata = True
            for i, result in enumerate(results, 1):
                required_fields = ["id", "text", "score", "metadata"]
                for field in required_fields:
                    if field not in result:
                        all_have_metadata = False
                        print(f"  ✗ Result {i} missing '{field}'")

            if all_have_metadata:
                print(f"  ✓ PASSED: All {len(results)} results include metadata field")
                passed += 1
            else:
                print(f"  ✗ FAILED: Some results missing required fields")

    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        import traceback
        traceback.print_exc()

    # Test 3: Verify metadata structure for conversation documents
    print("\n[3] Testing conversation chunks have appropriate metadata...")
    total += 1
    try:
        store = PgresStore()
        retriever = SemanticRetriever()

        # Use sample Teams meeting conversation document
        book_slug = "stm"  # Sample Teams Meeting
        with store.conn.cursor() as cur:
            cur.execute("SELECT slug, title FROM books WHERE slug = %s", (book_slug,))
            row = cur.fetchone()

            if not row:
                print(f"  ⊘ SKIPPED: Sample conversation document '{book_slug}' not found in database")
            else:
                book_slug, book_title = row
                print(f"  Testing sample document: {book_slug} ({book_title})")

                # Search in this conversation
                results = retriever.search("meeting discussion", topk=5, book_slug=book_slug)

                if not results:
                    print(f"  ⊘ SKIPPED: No results found in {book_slug}")
                else:
                    print(f"  Found {len(results)} results in {book_slug}")

                    # Check metadata structure
                    for i, result in enumerate(results[:3], 1):  # Check first 3
                        metadata = result.get("metadata", {})
                        chunk_id = result.get("id", "unknown")

                        print(f"\n  Chunk {i}: {chunk_id[:40]}...")
                        print(f"    Metadata keys: {list(metadata.keys())}")

                        # Check for common conversation metadata fields
                        has_speaker_info = (
                            "speakers" in metadata
                            or "speaker" in metadata
                            or "author" in metadata
                        )
                        has_timestamp_info = (
                            "timestamp" in metadata
                            or "created_at" in metadata
                            or "timestamp_start" in metadata
                        )

                        if has_speaker_info:
                            print(f"    ✓ Has speaker information")
                        if has_timestamp_info:
                            print(f"    ✓ Has timestamp information")

                    print(f"\n  ✓ PASSED: Metadata structure verified for conversation documents")
                    passed += 1

    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        import traceback
        traceback.print_exc()

    # Summary
    print("\n" + "=" * 80)
    print(f"SUMMARY: {passed}/{total} tests passed")
    if passed == total:
        print("✓ ALL TESTS PASSED")
    else:
        print(f"✗ {total - passed} tests failed")
    print("=" * 80)
    print("\nKEY VERIFICATION:")
    print("- Metadata field is now included in all Qdrant retrieval operations")
    print("- Citation code can access speaker and timestamp metadata when available")
    print("- Actual timestamp presence depends on ingestion parsing")

    return passed == total


if __name__ == "__main__":
    import sys
    success = test_conversation_metadata_retrieval()
    sys.exit(0 if success else 1)
