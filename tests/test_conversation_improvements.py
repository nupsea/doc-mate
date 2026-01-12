"""
Quick tests for conversation search improvements.
"""

from datetime import datetime, timedelta
from src.flows.book_query import _diversify_conversation_results, _parse_timestamp


def test_all():
    """Run all tests."""
    print("=" * 80)
    print("CONVERSATION SEARCH IMPROVEMENT TESTS")
    print("=" * 80)

    passed = 0
    total = 0

    # Test 1: Timestamp parsing
    print("\n[1] Testing timestamp parsing...")
    total += 1
    try:
        assert _parse_timestamp("2024-01-15 14:30:00") is not None
        assert _parse_timestamp("2024-01-15 14:30") is not None
        assert _parse_timestamp("2024-01-15") is not None
        assert _parse_timestamp(None) is None
        assert _parse_timestamp("invalid") is None

        dt = _parse_timestamp("2024-01-15 14:30:00")
        assert dt.year == 2024 and dt.hour == 14 and dt.minute == 30
        print("  ✓ PASSED")
        passed += 1
    except AssertionError as e:
        print(f"  ✗ FAILED: {e}")

    # Test 2: Temporal spreading
    print("\n[2] Testing temporal diversity (5min min gap)...")
    total += 1
    try:
        base_time = datetime(2024, 1, 15, 14, 0, 0)
        chunks = []
        for i in range(10):
            chunks.append({
                "id": f"chunk_{i}",
                "text": f"Text {i}",
                "metadata": {
                    "timestamp": (base_time + timedelta(minutes=i)).strftime("%Y-%m-%d %H:%M:%S")
                }
            })

        diversified = _diversify_conversation_results(chunks, target_count=5)
        assert len(diversified) <= 5, f"Should return <=5 results, got {len(diversified)}"

        # Check temporal spacing
        if len(diversified) >= 2:
            timestamps = [_parse_timestamp(c["metadata"]["timestamp"]) for c in diversified]
            for i in range(len(timestamps) - 1):
                gap_seconds = abs((timestamps[i+1] - timestamps[i]).total_seconds())
                assert gap_seconds >= 300, f"Gap too small: {gap_seconds}s between {i} and {i+1}"

        print(f"  ✓ PASSED (diversified {len(chunks)} → {len(diversified)} results)")
        passed += 1
    except AssertionError as e:
        print(f"  ✗ FAILED: {e}")

    # Test 3: Speaker balancing
    print("\n[3] Testing speaker balancing (max 2 per speaker)...")
    total += 1
    try:
        chunks = []
        # 8 from Alice
        for i in range(8):
            chunks.append({
                "id": f"alice_{i}",
                "text": f"Alice {i}",
                "metadata": {
                    "speaker": "Alice",
                    "timestamp": f"2024-01-15 14:{i:02d}:00"
                }
            })
        # 2 from Bob
        for i in range(2):
            chunks.append({
                "id": f"bob_{i}",
                "text": f"Bob {i}",
                "metadata": {
                    "speaker": "Bob",
                    "timestamp": f"2024-01-15 15:{i:02d}:00"
                }
            })

        diversified = _diversify_conversation_results(chunks, target_count=5)
        alice_count = sum(1 for c in diversified if c["metadata"]["speaker"] == "Alice")
        bob_count = sum(1 for c in diversified if c["metadata"]["speaker"] == "Bob")

        assert alice_count <= 2, f"Too many Alice: {alice_count}"
        assert bob_count <= 2, f"Too many Bob: {bob_count}"
        print(f"  ✓ PASSED (Alice: {alice_count}, Bob: {bob_count})")
        passed += 1
    except AssertionError as e:
        print(f"  ✗ FAILED: {e}")

    # Test 4: Small inputs preserved
    print("\n[4] Testing small input preservation (≤3 chunks)...")
    total += 1
    try:
        chunks = [
            {"id": "1", "text": "Text 1", "metadata": {}},
            {"id": "2", "text": "Text 2", "metadata": {}},
            {"id": "3", "text": "Text 3", "metadata": {}},
        ]

        diversified = _diversify_conversation_results(chunks)
        assert len(diversified) == 3, f"Should preserve all 3, got {len(diversified)}"
        print("  ✓ PASSED")
        passed += 1
    except AssertionError as e:
        print(f"  ✗ FAILED: {e}")

    # Test 5: No metadata handling
    print("\n[5] Testing graceful handling of missing metadata...")
    total += 1
    try:
        chunks = [{"id": str(i), "text": f"Text {i}", "metadata": {}} for i in range(10)]
        diversified = _diversify_conversation_results(chunks, target_count=5)
        assert 0 < len(diversified) <= 5, f"Should return 1-5 results, got {len(diversified)}"
        print(f"  ✓ PASSED (got {len(diversified)} results)")
        passed += 1
    except AssertionError as e:
        print(f"  ✗ FAILED: {e}")

    # Test 6: Mixed metadata
    print("\n[6] Testing mixed metadata (some have timestamps/speakers, some don't)...")
    total += 1
    try:
        chunks = [
            {"id": "1", "text": "Text 1", "metadata": {"timestamp": "2024-01-15 14:00:00", "speaker": "Alice"}},
            {"id": "2", "text": "Text 2", "metadata": {}},
            {"id": "3", "text": "Text 3", "metadata": {"timestamp": "2024-01-15 14:10:00"}},
            {"id": "4", "text": "Text 4", "metadata": {"speaker": "Bob"}},
            {"id": "5", "text": "Text 5", "metadata": {"timestamp": "2024-01-15 14:20:00", "speaker": "Alice"}},
        ]

        diversified = _diversify_conversation_results(chunks, target_count=3)
        assert 0 < len(diversified) <= 3, f"Should return 1-3 results, got {len(diversified)}"
        print(f"  ✓ PASSED (got {len(diversified)} results)")
        passed += 1
    except AssertionError as e:
        print(f"  ✗ FAILED: {e}")

    # Summary
    print("\n" + "=" * 80)
    print(f"SUMMARY: {passed}/{total} tests passed")
    if passed == total:
        print("✓ ALL TESTS PASSED")
    else:
        print(f"✗ {total - passed} tests failed")
    print("=" * 80)

    return passed == total


if __name__ == "__main__":
    import sys
    success = test_all()
    sys.exit(0 if success else 1)
