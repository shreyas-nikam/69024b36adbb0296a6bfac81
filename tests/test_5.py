import pytest
import pandas as pd
import numpy as np
from definition_8e59ca783b694698851f93ef0eafddb8 import ConsumerCreditScoringModel

# Instantiate the model class for testing the instance method
model_instance = ConsumerCreditScoringModel()

def test_transform_features_typical_inputs():
    """
    Test _transform_features with a set of typical, valid inputs.
    Verifies all four transformations (log, standardize, clip, none) work as expected.
    """
    X = pd.DataFrame([{
        'income': 75000,
        'credit_score': 720,
        'dti': 0.25,
        'tenure': 84
    }])
    transformed_X = model_instance._transform_features(X)

    expected_log_income = np.log(75000)
    expected_credit_score_scaled = (720 - 680) / 100  # (score - mean) / std
    expected_dti_clipped = 0.25  # Below clipping threshold
    expected_tenure = 84

    assert np.isclose(transformed_X[0, 0], expected_log_income)
    assert np.isclose(transformed_X[0, 1], expected_credit_score_scaled)
    assert np.isclose(transformed_X[0, 2], expected_dti_clipped)
    assert np.isclose(transformed_X[0, 3], expected_tenure)

@pytest.mark.parametrize("income_input, expected_log_income", [
    (0, np.log(1)),      # Edge: income=0, should be treated as 1 for log
    (0.5, np.log(1)),    # Edge: income < 1, should be treated as 1 for log
    (1, np.log(1)),      # Edge: income=1, should be treated as 1 for log
    (1000000, np.log(1000000)) # Large income
])
def test_transform_features_income_edge_cases(income_input, expected_log_income):
    """
    Test the logarithmic transformation for income, including edge cases
    where income is very low or zero, ensuring `np.maximum(X['income'], 1)` logic.
    """
    X = pd.DataFrame([{
        'income': income_input,
        'credit_score': 680,
        'dti': 0.1,
        'tenure': 12
    }])
    transformed_X = model_instance._transform_features(X)
    assert np.isclose(transformed_X[0, 0], expected_log_income)

@pytest.mark.parametrize("dti_input, expected_dti_clipped", [
    (0.0, 0.0),      # Edge: DTI at minimum
    (0.649, 0.649),  # Just below clipping threshold
    (0.65, 0.65),    # Edge: DTI at clipping threshold
    (1.2, 0.65),     # Above clipping threshold
    (1.5, 0.65)      # Edge: DTI at maximum allowed domain value, but clipped
])
def test_transform_features_dti_clipping_edge_cases(dti_input, expected_dti_clipped):
    """
    Test the clipping transformation for debt-to-income ratio (DTI),
    covering values below, at, and above the clipping threshold of 0.65.
    """
    X = pd.DataFrame([{
        'income': 50000,
        'credit_score': 680,
        'dti': dti_input,
        'tenure': 12
    }])
    transformed_X = model_instance._transform_features(X)
    assert np.isclose(transformed_X[0, 2], expected_dti_clipped)

@pytest.mark.parametrize("credit_score_input, expected_credit_score_scaled", [
    (300, (300 - 680) / 100), # Edge: Minimum credit score
    (850, (850 - 680) / 100), # Edge: Maximum credit score
    (680, (680 - 680) / 100), # Credit score at the mean (should scale to 0)
    (550, (550 - 680) / 100)  # Below average score
])
def test_transform_features_credit_score_standardization_edge_cases(credit_score_input, expected_credit_score_scaled):
    """
    Test the standardization for credit score, including the minimum, maximum,
    and mean values based on the specification (mean=680, std=100).
    """
    X = pd.DataFrame([{
        'income': 50000,
        'credit_score': credit_score_input,
        'dti': 0.1,
        'tenure': 12
    }])
    transformed_X = model_instance._transform_features(X)
    assert np.isclose(transformed_X[0, 1], expected_credit_score_scaled)

@pytest.mark.parametrize("missing_column_key", [
    'income',
    'credit_score',
    'dti',
    'tenure'
])
def test_transform_features_missing_columns_in_dataframe(missing_column_key):
    """
    Test that a KeyError is raised if the input DataFrame `X` is missing
    any of the required feature columns.
    """
    data = {
        'income': [50000],
        'credit_score': [700],
        'dti': [0.3],
        'tenure': [60]
    }
    # Remove one column for the test
    del data[missing_column_key]
    X_missing = pd.DataFrame(data)

    with pytest.raises(KeyError, match=f"'{missing_column_key}'"):
        model_instance._transform_features(X_missing)