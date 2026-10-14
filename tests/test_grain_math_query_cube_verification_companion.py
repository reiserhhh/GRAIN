"""HM32 companion normalizes serialization, not numerical claims or roles."""

import pytest

from scripts.run_grain_math_query_cube_verification_companion import normalize


def test_only_native_booleans_are_converted_without_mutation():
    original = [
        {
            "inherited": "True",
            "greedy_first_token_is_candidate": "True",
            "complete_response_parse_rate_measured": "False",
            "conditional_score": "0.123",
        }
    ]
    converted = normalize(original)
    assert converted == [
        {
            "inherited": "True",
            "greedy_first_token_is_candidate": "1",
            "complete_response_parse_rate_measured": "0",
            "conditional_score": "0.123",
        }
    ]
    assert original[0]["greedy_first_token_is_candidate"] == "True"


def test_unexpected_bool_encoding_is_not_silently_accepted():
    with pytest.raises(ValueError, match="changed type"):
        normalize([{"greedy_first_token_is_candidate": "unknown"}])
