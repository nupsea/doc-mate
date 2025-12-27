"""
Edge case tests for local LLM quirks and robustness.

Tests specific issues encountered with Ollama/local models:
1. Parameter type inconsistencies (list vs string)
2. JSON parsing edge cases
3. Function calling format variations
4. Response length handling
5. Conditional prompting based on doc types
"""
import asyncio
from src.mcp_client.agent import BookMateAgent
from unittest.mock import MagicMock, AsyncMock, patch


async def test_parameter_normalization():
    """Test book_identifier normalization for list vs string."""
    print("\n" + "=" * 80)
    print("TEST: Parameter Normalization (List vs String)")
    print("=" * 80)

    agent = BookMateAgent(provider="local")

    test_cases = [
        {
            "input": {"book_identifier": "stm"},
            "expected": "stm",
            "description": "String input (normal case)"
        },
        {
            "input": {"book_identifier": ["stm"]},
            "expected": "stm",
            "description": "List with single element (Ollama quirk)"
        },
        {
            "input": {"book_identifier": ["mam", "ili"]},
            "expected": "mam",
            "description": "List with multiple elements (take first)"
        },
        {
            "input": {"book_identifiers": ["mam", "ili", "ody"]},
            "expected": ["mam", "ili", "ody"],
            "description": "Multiple books (should remain list)"
        },
        {
            "input": {"book_identifiers": "['mam', 'stm']"},
            "expected": ["mam", "stm"],
            "description": "String representation of list (Python literal)"
        },
        {
            "input": {"book_identifiers": '["mam", "stm"]'},
            "expected": ["mam", "stm"],
            "description": "JSON string representation of list"
        },
    ]

    passed = 0
    failed = 0

    for test in test_cases:
        print(f"\n{test['description']}")
        print(f"  Input: {test['input']}")

        try:
            result_args, _ = agent._translate_book_identifier(test['input'].copy())

            if "book_identifier" in test['input']:
                actual = result_args.get("book_identifier")
            else:
                actual = result_args.get("book_identifiers")

            if actual == test['expected']:
                print(f"  Result: {actual}")
                print(f"  Status: PASS")
                passed += 1
            else:
                print(f"  Expected: {test['expected']}")
                print(f"  Got: {actual}")
                print(f"  Status: FAIL")
                failed += 1

        except Exception as e:
            print(f"  Error: {e}")
            print(f"  Status: ERROR")
            failed += 1

    print(f"\nSummary: {passed} passed, {failed} failed")
    print("=" * 80)


async def test_conditional_prompting():
    """Test that prompts are built correctly based on doc types."""
    print("\n" + "=" * 80)
    print("TEST: Conditional Prompting Based on Doc Types")
    print("=" * 80)

    from src.mcp_client.prompts import get_system_prompt

    test_cases = [
        {
            "doc_types": {"book"},
            "should_include": ["COMPARATIVE ANALYSIS"],
            "should_exclude": ["SPEAKER ATTRIBUTION"],
            "description": "Books only"
        },
        {
            "doc_types": {"script", "conversation"},
            "should_include": ["SPEAKER ATTRIBUTION"],
            "should_exclude": ["COMPARATIVE ANALYSIS"],
            "description": "Scripts/conversations only"
        },
        {
            "doc_types": {"book", "script"},
            "should_include": ["COMPARATIVE ANALYSIS", "SPEAKER ATTRIBUTION"],
            "should_exclude": [],
            "description": "Mixed doc types"
        },
        {
            "doc_types": {"tech_doc"},
            "should_include": ["TECHNICAL DOCUMENTATION"],
            "should_exclude": ["SPEAKER ATTRIBUTION", "COMPARATIVE ANALYSIS"],
            "description": "Tech docs only"
        },
    ]

    passed = 0
    failed = 0

    for test in test_cases:
        print(f"\n{test['description']}")
        print(f"  Doc types: {test['doc_types']}")

        try:
            prompt = get_system_prompt("Test docs", doc_types=test['doc_types'], use_simple=False)

            all_good = True

            for should_have in test['should_include']:
                if should_have not in prompt:
                    print(f"  FAIL: Missing '{should_have}'")
                    all_good = False

            for should_not_have in test['should_exclude']:
                if should_not_have in prompt:
                    print(f"  FAIL: Should not include '{should_not_have}'")
                    all_good = False

            if all_good:
                print(f"  Status: PASS")
                passed += 1
            else:
                print(f"  Status: FAIL")
                failed += 1

        except Exception as e:
            print(f"  Error: {e}")
            print(f"  Status: ERROR")
            failed += 1

    print(f"\nSummary: {passed} passed, {failed} failed")
    print("=" * 80)


async def test_temperature_settings():
    """Test that local LLM uses correct temperature settings."""
    print("\n" + "=" * 80)
    print("TEST: Temperature and Max Tokens Settings")
    print("=" * 80)

    test_cases = [
        {
            "provider": "local",
            "expected_temp": 0.0,
            "expected_max_tokens": 1536,
            "description": "Local LLM (Ollama)"
        },
        {
            "provider": "openai",
            "expected_temp": 0.1,
            "expected_max_tokens": 4096,
            "description": "OpenAI"
        },
    ]

    passed = 0
    failed = 0

    for test in test_cases:
        print(f"\n{test['description']}")

        try:
            agent = BookMateAgent(provider=test['provider'])

            actual_temp = agent._get_temperature_for_provider()
            actual_max_tokens = agent._get_max_tokens_for_provider()

            temp_match = actual_temp == test['expected_temp']
            tokens_match = actual_max_tokens == test['expected_max_tokens']

            print(f"  Temperature: {actual_temp} (expected: {test['expected_temp']}) - {'PASS' if temp_match else 'FAIL'}")
            print(f"  Max tokens: {actual_max_tokens} (expected: {test['expected_max_tokens']}) - {'PASS' if tokens_match else 'FAIL'}")

            if temp_match and tokens_match:
                print(f"  Status: PASS")
                passed += 1
            else:
                print(f"  Status: FAIL")
                failed += 1

        except Exception as e:
            print(f"  Error: {e}")
            print(f"  Status: ERROR")
            failed += 1

    print(f"\nSummary: {passed} passed, {failed} failed")
    print("=" * 80)


async def test_simple_prompt():
    """Test that local LLM uses simplified prompt."""
    print("\n" + "=" * 80)
    print("TEST: Simplified Prompt for Local LLM")
    print("=" * 80)

    from src.mcp_client.prompts import get_system_prompt

    # Simple prompt test
    simple_prompt = get_system_prompt("Test docs", doc_types={"book"}, use_simple=True)
    full_prompt = get_system_prompt("Test docs", doc_types={"book"}, use_simple=False)

    print(f"\nSimple prompt length: {len(simple_prompt)} chars")
    print(f"Full prompt length: {len(full_prompt)} chars")
    print(f"Reduction: {100 - (len(simple_prompt) * 100 // len(full_prompt))}%")

    if len(simple_prompt) < len(full_prompt):
        print("\nStatus: PASS - Simple prompt is shorter")
    else:
        print("\nStatus: FAIL - Simple prompt should be shorter")

    print(f"\nSimple prompt content:")
    print(simple_prompt[:300] + "...")

    print("=" * 80)


async def run_all_edge_case_tests():
    """Run all edge case tests."""
    print("\n" + "=" * 80)
    print("LOCAL LLM EDGE CASE TEST SUITE")
    print("=" * 80)

    await test_parameter_normalization()
    await test_conditional_prompting()
    await test_temperature_settings()
    await test_simple_prompt()

    print("\n" + "=" * 80)
    print("ALL EDGE CASE TESTS COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(run_all_edge_case_tests())
