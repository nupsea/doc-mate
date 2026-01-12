"""
Test chat export format parsing with date/time timestamps.
"""

import re


def test_chat_export_pattern():
    """Test that chat export format with date/time is correctly parsed."""
    print("=" * 80)
    print("CHAT EXPORT FORMAT PARSING TEST")
    print("=" * 80)

    # Chat export pattern from conversation_parser.py
    chat_pattern = r'^\s*\[(\d{1,2}/\d{1,2}/\d{4},\s*\d{1,2}:\s*\d{2}:\s*\d{2}\s*[ap]m)\]\s*([^:\[\]]+):\s*(.+)$'

    # Test cases with generic names including edge cases
    test_lines = [
        "[20/9/2019, 10: 52:23 am] Alice: Let's meet at the office on Sunday.",
        "[20/9/2019, 11: 22:31 am] Bob: Sounds good!",
        "[21/9/2019, 11: 24:29 pm] Alice: I met with Carol for lunch today.",
        "[29/5/2024, 6: 30:15 pm] David: Test message with longer text here.",
        "[1/1/2024, 9: 05:00 am] Eve: Morning message on single digit day.",
        "  [30/5/2024, 3: 45:12 pm] Frank: Message with leading whitespace.",
        "[2/6/2024, 12: 00:00 pm] Grace Ann: Speaker with space in name.",
    ]

    print("\n[1] Testing chat export pattern matching...")

    passed = 0
    for i, line in enumerate(test_lines, 1):
        match = re.match(chat_pattern, line)

        if match:
            groups = match.groups()
            timestamp = groups[0]
            speaker = groups[1]
            message = groups[2]

            print(f"\n  Test {i}: ✓ MATCHED")
            print(f"    Timestamp: {timestamp}")
            print(f"    Speaker: {speaker}")
            print(f"    Message: {message[:40]}...")

            # Verify structure
            assert len(groups) == 3, f"Expected 3 groups, got {len(groups)}"
            assert timestamp, "Timestamp should not be empty"
            assert speaker, "Speaker should not be empty"
            assert message, "Message should not be empty"
            assert "/" in timestamp, "Timestamp should contain date separator"
            assert ":" in timestamp, "Timestamp should contain time separator"

            passed += 1
        else:
            print(f"\n  Test {i}: ✗ FAILED - Pattern did not match")
            print(f"    Line: {line}")

    print("\n" + "=" * 80)
    print(f"RESULTS: {passed}/{len(test_lines)} tests passed")

    if passed == len(test_lines):
        print("✓ ALL CHAT EXPORT FORMAT TESTS PASSED")
        print("\nThe pattern correctly extracts:")
        print("  - Full timestamp (date + time)")
        print("  - Speaker name")
        print("  - Message text")
        print("\nRe-ingesting will now extract proper speaker names from chat exports!")
    else:
        print(f"✗ {len(test_lines) - passed} tests failed")

    print("=" * 80)

    return passed == len(test_lines)


if __name__ == "__main__":
    import sys
    success = test_chat_export_pattern()
    sys.exit(0 if success else 1)
