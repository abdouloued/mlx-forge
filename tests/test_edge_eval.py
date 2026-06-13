import pytest
from recipes.edge_android.scorer import score_response, parse_answer


def test_parse_finds_expected_keyword():
    assert parse_answer("The battery life is about 12 hours.", keywords=["12 hours"]) is True


def test_parse_case_insensitive():
    assert parse_answer("BATTERY LIFE IS 12 HOURS", keywords=["12 hours"]) is True


def test_parse_no_match_returns_false():
    assert parse_answer("I don't know.", keywords=["12 hours"]) is False


def test_parse_any_keyword_matches():
    assert parse_answer("Twelve hours battery.", keywords=["12 hours", "twelve hours"]) is True


def test_perfect_score():
    example = {
        "expected_keywords": ["12 hours"],
        "max_words": 20,
    }
    response = "The battery lasts 12 hours."
    assert score_response(response, example) == 1.0


def test_correct_but_verbose_loses_conciseness():
    example = {
        "expected_keywords": ["12 hours"],
        "max_words": 5,
    }
    response = "The battery life of this device is approximately 12 hours under normal usage conditions."
    score = score_response(response, example)
    assert 0.0 < score < 1.0


def test_wrong_answer_zero_correctness():
    example = {
        "expected_keywords": ["12 hours"],
        "max_words": 20,
    }
    response = "The battery lasts 6 hours."
    score = score_response(response, example)
    assert score < 0.6  # no correctness credit


def test_empty_response_zero_score():
    example = {"expected_keywords": ["12 hours"], "max_words": 20}
    assert score_response("", example) == 0.0
