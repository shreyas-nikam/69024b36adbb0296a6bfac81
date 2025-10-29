import pytest
from definition_f533b4b73866460481a562a0fd36551f import ConsumerCreditScoringModel

@pytest.fixture(scope="module")
def model_instance():
    """Provides a ConsumerCreditScoringModel instance for tests."""
    return ConsumerCreditScoringModel()

def test_validate_inputs_valid_cases(model_instance):
    """
    Test with various valid inputs within the specified business ranges.
    No ValueError should be raised.
    """
    # Test case 1: Typical valid inputs
    model_instance._validate_inputs(income=75000, credit_score=720, dti=0.45, tenure=60)
    # Test case 2: Valid inputs at the lower bounds
    model_instance._validate_inputs(income=1, credit_score=300, dti=0.0, tenure=0)
    # Test case 3: Valid inputs at the upper bounds (where applicable)
    model_instance._validate_inputs(income=1_000_000, credit_score=850, dti=1.5, tenure=600) # tenure has no explicit upper bound beyond practical limits, so a large positive is fine
    # If any of the above raised a ValueError, pytest.fail would not be reached.

@pytest.mark.parametrize("income, expected_error_msg", [
    (0, "income must be positive, got 0"),       # Edge case: zero income
    (-100, "income must be positive, got -100"), # Negative income
])
def test_validate_inputs_invalid_income(model_instance, income, expected_error_msg):
    """
    Test cases for invalid 'income' values (non-positive).
    Expects ValueError with a specific message.
    """
    with pytest.raises(ValueError, match=expected_error_msg):
        model_instance._validate_inputs(income=income, credit_score=700, dti=0.5, tenure=24)

@pytest.mark.parametrize("credit_score, expected_error_msg", [
    (299, "credit_score must be 300-850, got 299"), # Just below lower bound
    (851, "credit_score must be 300-850, got 851"), # Just above upper bound
    (0, "credit_score must be 300-850, got 0"),     # Far below lower bound
    (1000, "credit_score must be 300-850, got 1000"), # Far above upper bound
])
def test_validate_inputs_invalid_credit_score(model_instance, credit_score, expected_error_msg):
    """
    Test cases for invalid 'credit_score' values (outside 300-850 range).
    Expects ValueError with a specific message.
    """
    with pytest.raises(ValueError, match=expected_error_msg):
        model_instance._validate_inputs(income=50000, credit_score=credit_score, dti=0.5, tenure=24)

@pytest.mark.parametrize("dti, expected_error_msg", [
    (-0.1, "dti must be 0-1.5, got -0.1"),   # Just below lower bound
    (1.51, "dti must be 0-1.5, got 1.51"),   # Just above upper bound
    (-1.0, "dti must be 0-1.5, got -1.0"),   # Far below lower bound
    (2.0, "dti must be 0-1.5, got 2.0"),     # Far above upper bound
])
def test_validate_inputs_invalid_dti(model_instance, dti, expected_error_msg):
    """
    Test cases for invalid 'dti' values (outside 0-1.5 range).
    Expects ValueError with a specific message.
    """
    with pytest.raises(ValueError, match=expected_error_msg):
        model_instance._validate_inputs(income=50000, credit_score=700, dti=dti, tenure=24)

@pytest.mark.parametrize("tenure, expected_error_msg", [
    (-1, "tenure must be non-negative, got -1"),     # Just below lower bound
    (-100, "tenure must be non-negative, got -100"), # Negative tenure
])
def test_validate_inputs_invalid_tenure(model_instance, tenure, expected_error_msg):
    """
    Test cases for invalid 'tenure' values (negative).
    Expects ValueError with a specific message.
    """
    with pytest.raises(ValueError, match=expected_error_msg):
        model_instance._validate_inputs(income=50000, credit_score=700, dti=0.5, tenure=tenure)