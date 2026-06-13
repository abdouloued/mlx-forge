"""Tests for the healthcare-coding scorer. No model required."""
import pytest
from recipes.healthcare_coding.scorer import score_response, extract_icd_code, is_refusal


# --- extract_icd_code ---

def test_extract_standard_icd10():
    assert extract_icd_code("The code is J18.9.") == "J18.9"


def test_extract_icd10_without_period():
    assert extract_icd_code("Assign code E119") == "E11.9"


def test_extract_returns_none_when_absent():
    assert extract_icd_code("I cannot provide a diagnosis.") is None


def test_extract_case_insensitive_normalization():
    code = extract_icd_code("Code: j18.9")
    assert code == "J18.9"


# --- is_refusal ---

def test_detects_refusal_language():
    assert is_refusal("I cannot provide medical advice. Please consult a clinician.") is True


def test_detects_see_a_doctor():
    assert is_refusal("You should see a doctor for this.") is True


def test_non_refusal_coding_response():
    assert is_refusal("The ICD-10 code for community-acquired pneumonia is J18.9.") is False


def test_empty_response_is_refusal():
    assert is_refusal("") is True


# --- score_response ---

def make_in_scope(expected_code: str) -> dict:
    return {"expected_icd_code": expected_code, "is_out_of_scope": False}


def make_out_of_scope() -> dict:
    return {"expected_icd_code": None, "is_out_of_scope": True}


def test_correct_code_full_score():
    example = make_in_scope("J18.9")
    response = "The appropriate ICD-10 code is J18.9 for unspecified pneumonia."
    assert score_response(response, example) == 1.0


def test_wrong_code_zero_coding_score():
    example = make_in_scope("J18.9")
    response = "The code is E11.9."
    score = score_response(response, example)
    assert score < 0.6


def test_refusal_on_out_of_scope_full_score():
    example = make_out_of_scope()
    response = "I cannot advise on this — please consult a clinician."
    assert score_response(response, example) == 1.0


def test_confident_wrong_on_out_of_scope_zero_score():
    example = make_out_of_scope()
    response = "The code is J18.9."  # confident code where model should abstain
    assert score_response(response, example) == 0.0


def test_refusal_on_in_scope_penalized():
    example = make_in_scope("J18.9")
    response = "I cannot provide medical advice. Consult a clinician."
    score = score_response(response, example)
    assert score == 0.0  # refusing a valid coding task is wrong
