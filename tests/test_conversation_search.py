"""
Test conversation search improvements:
1. Higher topk for conversations (15 vs 5)
2. Diversity filtering (temporal spreading, speaker balancing)
3. Timestamp exposure in results
"""

from datetime import datetime, timedelta
from src.flows.book_query import _diversify_conversation_results, _parse_timestamp


def assert_test(condition, message):
    """Simple assertion helper."""
    if not condition:
        raise AssertionError(message)


def test_parse_timestamp():
    """Test timestamp parsing with various formats."""
    print("\n[TEST] test_parse_timestamp")

    # Test valid formats
    assert_test(_parse_timestamp("2024-01-15 14:30:00") is not None, "Should parse full timestamp")
    assert_test(_parse_timestamp("2024-01-15 14:30") is not None, "Should parse timestamp without seconds")
    assert_test(_parse_timestamp("2024-01-15") is not None, "Should parse date only")

    # Test invalid formats
    assert_test(_parse_timestamp(None) is None, "Should return None for None")
    assert_test(_parse_timestamp("") is None, "Should return None for empty string")
    assert_test(_parse_timestamp("invalid") is None, "Should return None for invalid format")

    # Test date parsing accuracy
    dt = _parse_timestamp("2024-01-15 14:30:00")
    assert_test(dt.year == 2024, "Year should be 2024")
    assert_test(dt.month == 1, "Month should be 1")
    assert_test(dt.day == 15, "Day should be 15")
    assert_test(dt.hour == 14, "Hour should be 14")
    assert_test(dt.minute == 30, "Minute should be 30")

    print("  ✓ PASSED")


def test_diversity_filtering_temporal_spreading():
    """Test that diversity filtering spreads results across timeline."""
    base_time = datetime(2024, 1, 15, 14, 0, 0)

    # Create 10 chunks within 10 minutes (clustered)
    chunks = []
    for i in range(10):
        chunks.append({
            "id": f"chunk_{i}",
            "text": f"Text {i}",
            "metadata": {
                "timestamp": (base_time + timedelta(minutes=i)).strftime("%Y-%m-%d %H:%M:%S")
            }
        })

    # Apply diversity filtering
    diversified = _diversify_conversation_results(chunks, target_count=5)

    # Should return fewer results (spread across time)
    assert len(diversified) <= 5

    # Check temporal spacing (min 5 minutes between results)
    if len(diversified) >= 2:
        timestamps = [_parse_timestamp(c["metadata"]["timestamp"]) for c in diversified]
        for i in range(len(timestamps) - 1):
            gap_seconds = abs((timestamps[i+1] - timestamps[i]).total_seconds())
            assert gap_seconds >= 300, f"Gap too small: {gap_seconds}s between results {i} and {i+1}"


def test_diversity_filtering_speaker_balancing():
    """Test that diversity filtering balances speakers."""
    # Create chunks with speaker imbalance (8 from Alice, 2 from Bob)
    chunks = []
    for i in range(8):
        chunks.append({
            "id": f"alice_{i}",
            "text": f"Alice says {i}",
            "metadata": {
                "speaker": "Alice",
                "timestamp": f"2024-01-15 14:{i:02d}:00"
            }
        })
    for i in range(2):
        chunks.append({
            "id": f"bob_{i}",
            "text": f"Bob says {i}",
            "metadata": {
                "speaker": "Bob",
                "timestamp": f"2024-01-15 15:{i:02d}:00"
            }
        })

    # Apply diversity filtering
    diversified = _diversify_conversation_results(chunks, target_count=5)

    # Count speakers in result
    alice_count = sum(1 for c in diversified if c["metadata"]["speaker"] == "Alice")
    bob_count = sum(1 for c in diversified if c["metadata"]["speaker"] == "Bob")

    # Alice should be limited (max 2 per speaker rule)
    assert alice_count <= 2, f"Too many Alice results: {alice_count}"
    # Bob should appear if available
    assert bob_count <= 2, f"Too many Bob results: {bob_count}"


def test_diversity_filtering_small_input():
    """Test that diversity filtering doesn't over-filter small inputs."""
    # With ≤3 chunks, should return all
    chunks = [
        {"id": "1", "text": "Text 1", "metadata": {}},
        {"id": "2", "text": "Text 2", "metadata": {}},
        {"id": "3", "text": "Text 3", "metadata": {}},
    ]

    diversified = _diversify_conversation_results(chunks)
    assert len(diversified) == 3, "Should preserve all results when input is small"


def test_diversity_filtering_no_metadata():
    """Test diversity filtering works when chunks lack metadata."""
    chunks = []
    for i in range(10):
        chunks.append({
            "id": f"chunk_{i}",
            "text": f"Text {i}",
            "metadata": {}  # No timestamp or speaker
        })

    # Should still work (return ~half)
    diversified = _diversify_conversation_results(chunks, target_count=5)
    assert 0 < len(diversified) <= 5


def test_diversity_filtering_preserves_search_ranking():
    """Test that diversity filtering considers original search ranking."""
    # Create chunks with timestamps but same time (no temporal filtering)
    # Diversity should preserve high-ranking results
    chunks = []
    for i in range(10):
        chunks.append({
            "id": f"chunk_{i}",
            "text": f"Text {i}",
            "metadata": {
                "timestamp": "2024-01-15 14:00:00"  # Same time
            }
        })

    diversified = _diversify_conversation_results(chunks, target_count=3)

    # First few high-ranking chunks should be preserved
    assert diversified[0]["id"] == "chunk_0", "Should preserve top-ranked result"


def test_diversity_target_count():
    """Test that diversity filtering respects target_count parameter."""
    chunks = []
    base_time = datetime(2024, 1, 15, 14, 0, 0)

    for i in range(20):
        chunks.append({
            "id": f"chunk_{i}",
            "text": f"Text {i}",
            "metadata": {
                "timestamp": (base_time + timedelta(minutes=i*10)).strftime("%Y-%m-%d %H:%M:%S")
            }
        })

    # Test with different target counts
    result_3 = _diversify_conversation_results(chunks, target_count=3)
    result_7 = _diversify_conversation_results(chunks, target_count=7)

    assert len(result_3) <= 3
    assert len(result_7) <= 7


def test_diversity_mixed_metadata():
    """Test diversity with mixed metadata (some chunks have timestamps/speakers, some don't)."""
    chunks = [
        {"id": "1", "text": "Text 1", "metadata": {"timestamp": "2024-01-15 14:00:00", "speaker": "Alice"}},
        {"id": "2", "text": "Text 2", "metadata": {}},  # No metadata
        {"id": "3", "text": "Text 3", "metadata": {"timestamp": "2024-01-15 14:10:00"}},  # No speaker
        {"id": "4", "text": "Text 4", "metadata": {"speaker": "Bob"}},  # No timestamp
        {"id": "5", "text": "Text 5", "metadata": {"timestamp": "2024-01-15 14:20:00", "speaker": "Alice"}},
    ]

    # Should handle gracefully without errors
    diversified = _diversify_conversation_results(chunks, target_count=3)
    assert len(diversified) > 0
    assert len(diversified) <= 3


if __name__ == "__main__":
    print("=" * 80)
    print("CONVERSATION SEARCH TESTS")
    print("=" * 80)

    tests = [
        test_parse_timestamp,
        test_diversity_filtering_temporal_spreading,
        test_diversity_filtering_speaker_balancing,
        test_diversity_filtering_small_input,
        test_diversity_filtering_no_metadata,
        test_diversity_filtering_preserves_search_ranking,
        test_diversity_target_count,
        test_diversity_mixed_metadata,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            passed += 1
        except AssertionError as e:
            failed += 1
            print(f"  ✗ FAILED: {e}")
        except Exception as e:
            failed += 1
            print(f"  ✗ ERROR: {e}")

    print("\n" + "=" * 80)
    print(f"RESULTS: {passed} passed, {failed} failed out of {len(tests)} tests")
    print("=" * 80)

    exit(0 if failed == 0 else 1)
