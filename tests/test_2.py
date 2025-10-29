import pytest
# definition_47ac2878bc5f4ba0918b3c0603d3cd52 block
from definition_47ac2878bc5f4ba0918b3c0603d3cd52 import ConsumerCreditScoringModel

@pytest.fixture(scope="module")
def model_instance():
    """
    Provides a ConsumerCreditScoringModel instance for tests.
    Assumes the ConsumerCreditScoringModel class is importable from definition_47ac2878bc5f4ba0918b3c0603d3cd52
    and its `predict` method, even if a stub, implements the input validation
    and returns a placeholder float for valid inputs, as detailed in the spec.
    """
    return ConsumerCreditScoringModel()

def test_predict_valid_inputs_returns_probability(model_instance):
    """
    Test that predict returns a float within [0, 1] for nominal valid inputs.
    Covers PROP-003 (probability_bounds) and basic expected functionality.
    """
    income = 50000.0
    credit_score = 700
    dti = 0.3
    tenure = 24
    
    prediction = model_instance.predict(income, credit_score, dti, tenure)
    
    assert isinstance(prediction, float)
    assert 0.0 <= prediction <= 1.0

def test_predict_edge_case_low_risk_valid(model_instance):
    """
    Test that predict returns a float within [0, 1] for valid inputs representing a very low-risk applicant.
    Covers PROP-003 and a valid edge case at the 'best' end of the input ranges.
    """
    income = 300000.0  # Very high income
    credit_score = 850  # Max FICO score
    dti = 0.05          # Very low DTI
    tenure = 600        # Very long tenure
    
    prediction = model_instance.predict(income, credit_score, dti, tenure)
    
    assert isinstance(prediction, float)
    assert 0.0 <= prediction <= 1.0

def test_predict_invalid_income_raises_error(model_instance):
    """
    Test that predict raises a ValueError for invalid (non-positive) income.
    Covers PROP-005 (input_validation_comprehensive) for income.
    """
    with pytest.raises(ValueError, match="income must be positive"):
        model_instance.predict(income=0.0, credit_score=700, dti=0.3, tenure=24)
    with pytest.raises(ValueError, match="income must be positive"):
        model_instance.predict(income=-100.0, credit_score=700, dti=0.3, tenure=24)

def test_predict_invalid_credit_score_raises_error(model_instance):
    """
    Test that predict raises a ValueError for credit_score outside the 300-850 range.
    Covers PROP-005 (input_validation_comprehensive) for credit_score.
    """
    with pytest.raises(ValueError, match="credit_score must be 300-850"):
        model_instance.predict(income=50000, credit_score=299, dti=0.3, tenure=24)
    with pytest.raises(ValueError, match="credit_score must be 300-850"):
        model_instance.predict(income=50000, credit_score=851, dti=0.3, tenure=24)

def test_predict_invalid_dti_raises_error(model_instance):
    """
    Test that predict raises a ValueError for dti outside the 0-1.5 range.
    Covers PROP-005 (input_validation_comprehensive) for dti.
    """
    with pytest.raises(ValueError, match="dti must be 0-1.5"):
        model_instance.predict(income=50000, credit_score=700, dti=-0.01, tenure=24)
    with pytest.raises(ValueError, match="dti must be 0-1.5"):
        model_instance.predict(income=50000, credit_score=700, dti=1.51, tenure=24)