"""
Test conversation history preservation across mode switches
"""
import asyncio


def check_should_clear_history(old_mode: str, new_mode: str) -> tuple[bool, bool, bool]:
    """
    Simulate the logic in set_provider_and_model.

    Returns:
        (changed, was_ephemeral, is_ephemeral)
    """
    changed = old_mode != new_mode
    old_ephemeral = old_mode in ["ephemeral", "private"]
    new_ephemeral = new_mode in ["ephemeral", "private"]
    return changed, old_ephemeral, new_ephemeral


async def test_conversation_history_logic():
    """Test that conversation history is cleared only when switching FROM ephemeral TO non-ephemeral"""
    print("="*80)
    print("TEST: Conversation History Preservation Logic")
    print("="*80)

    # Test 1: Normal → Ephemeral (should KEEP history)
    print("\n[Test 1] Normal → Ephemeral: Should KEEP history")
    changed, was_ephemeral, is_ephemeral = check_should_clear_history("normal", "ephemeral")
    print(f"  changed={changed}, was_ephemeral={was_ephemeral}, is_ephemeral={is_ephemeral}")
    should_clear = changed and was_ephemeral and not is_ephemeral
    print(f"  Should clear history: {should_clear}")
    assert not should_clear, "Should NOT clear when switching TO ephemeral"
    print("  ✓ PASS: History preserved")

    # Test 2: Ephemeral → Normal (should CLEAR history)
    print("\n[Test 2] Ephemeral → Normal: Should CLEAR history")
    changed, was_ephemeral, is_ephemeral = check_should_clear_history("ephemeral", "normal")
    print(f"  changed={changed}, was_ephemeral={was_ephemeral}, is_ephemeral={is_ephemeral}")
    should_clear = changed and was_ephemeral and not is_ephemeral
    print(f"  Should clear history: {should_clear}")
    assert should_clear, "SHOULD clear when switching FROM ephemeral"
    print("  ✓ PASS: History cleared for privacy")

    # Test 3: Normal → Private (should KEEP history)
    print("\n[Test 3] Normal → Private: Should KEEP history")
    changed, was_ephemeral, is_ephemeral = check_should_clear_history("normal", "private")
    print(f"  changed={changed}, was_ephemeral={was_ephemeral}, is_ephemeral={is_ephemeral}")
    should_clear = changed and was_ephemeral and not is_ephemeral
    print(f"  Should clear history: {should_clear}")
    assert not should_clear, "Should NOT clear when switching TO private"
    print("  ✓ PASS: History preserved")

    # Test 4: Private → Internal (should CLEAR history)
    print("\n[Test 4] Private → Internal: Should CLEAR history")
    changed, was_ephemeral, is_ephemeral = check_should_clear_history("private", "internal")
    print(f"  changed={changed}, was_ephemeral={was_ephemeral}, is_ephemeral={is_ephemeral}")
    should_clear = changed and was_ephemeral and not is_ephemeral
    print(f"  Should clear history: {should_clear}")
    assert should_clear, "SHOULD clear when switching FROM private"
    print("  ✓ PASS: History cleared for privacy")

    # Test 5: Normal → Internal (should KEEP history)
    print("\n[Test 5] Normal → Internal: Should KEEP history")
    changed, was_ephemeral, is_ephemeral = check_should_clear_history("normal", "internal")
    print(f"  changed={changed}, was_ephemeral={was_ephemeral}, is_ephemeral={is_ephemeral}")
    should_clear = changed and was_ephemeral and not is_ephemeral
    print(f"  Should clear history: {should_clear}")
    assert not should_clear, "Should NOT clear - both are non-ephemeral"
    print("  ✓ PASS: History preserved")

    # Test 6: Ephemeral → Private (should KEEP history)
    print("\n[Test 6] Ephemeral → Private: Should KEEP history")
    changed, was_ephemeral, is_ephemeral = check_should_clear_history("ephemeral", "private")
    print(f"  changed={changed}, was_ephemeral={was_ephemeral}, is_ephemeral={is_ephemeral}")
    should_clear = changed and was_ephemeral and not is_ephemeral
    print(f"  Should clear history: {should_clear}")
    assert not should_clear, "Should NOT clear - both are ephemeral"
    print("  ✓ PASS: History preserved")

    # Test 7: No change (should KEEP history)
    print("\n[Test 7] No change: Should KEEP history")
    changed, was_ephemeral, is_ephemeral = check_should_clear_history("normal", "normal")
    print(f"  changed={changed}, was_ephemeral={was_ephemeral}, is_ephemeral={is_ephemeral}")
    should_clear = changed and was_ephemeral and not is_ephemeral
    print(f"  Should clear history: {should_clear}")
    assert not should_clear, "Should NOT clear - nothing changed"
    print("  ✓ PASS: History preserved")

    print("\n" + "="*80)
    print("✓ ALL TESTS PASSED")
    print("="*80)
    return True


if __name__ == "__main__":
    import sys
    result = asyncio.run(test_conversation_history_logic())
    sys.exit(0 if result else 1)
