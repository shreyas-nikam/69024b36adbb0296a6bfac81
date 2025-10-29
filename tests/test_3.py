import pytest
# from definition_8e577cdadc704e759ca0442929dff00a import ConsumerCreditScoringModel

# --- Start of definition_8e577cdadc704e759ca0442929dff00a block ---
import math
import pandas as pd
import numpy as np

class ConsumerCreditScoringModel:
    def __init__(self):
        # Dummy values for internal components as a mock for testing
        self.model = None
        self.scaler = None
        self.explainer = None

    def _validate_inputs(self, income, credit_score, dti, tenure):
        if income <= 0:
            raise ValueError(f"income must be positive, got {income}")
        if not (300 <= credit_score <= 850):
            raise ValueError(f"credit_score must be 300-850, got {credit_score}")
        if not (0 <= dti <= 1.5):
            raise ValueError(f"dti must be 0-1.5, got {dti}")
        if tenure < 0:
            raise ValueError(f"tenure must be non-negative, got {tenure}")

    def predict(self, income: float, credit_score: int,
               dti: float, tenure: int) -> float:
        """
        A mock predict method for testing predict_with_decision.
        It simulates different default probabilities based on input ranges
        to hit various decision branches.
        """
        self._validate_inputs(income, credit_score, dti, tenure)

        # Simulate different probability ranges to trigger decision branches
        if income > 100000 and credit_score > 750 and dti < 0.2 and tenure > 60:
            return 0.05 # Low probability -> APPROVED
        elif income < 30000 or credit_score < 550 or dti > 0.6:
            return 0.25 # High probability -> DENIED
        elif (30000 <= income <= 100000) and (550 <= credit_score <= 750) and \
             (0.2 <= dti <= 0.6) and (12 <= tenure <= 60):
            return 0.15 # Medium probability -> REVIEW_REQUIRED
        else: # Fallback to a neutral value
            return 0.101 # A value that falls into REVIEW_REQUIRED by default

    def _generate_adverse_action_reasons(self, income, credit_score,
                                           dti, tenure, prob):
        """
        A mock method to generate specific adverse action reasons,
        mimicking the behavior described in the technical specification.
        """
        reasons = []
        # These thresholds are based on the example in the tech spec's
        # `_generate_adverse_action_reasons` method
        if credit_score < 640:
            reasons.append(f"Credit score {credit_score} below recommended minimum (640)")
        if income < 35000:
            reasons.append(f"Income ${income:,.0f} insufficient for loan amount")
        if dti > 0.43:
            reasons.append(f"Debt-to-income ratio {dti:.1%} exceeds guideline (43%)")
        if tenure < 12:
            reasons.append(f"Employment history {tenure} months below minimum (12)")

        if not reasons:
            reasons.append("Overall credit profile indicates high default risk")

        return reasons

    def predict_with_decision(self, income, credit_score, dti, tenure):
        """
        Predicts the default probability and provides a credit decision along with specific adverse action reasons if the application is denied. This function is essential for ECOA compliance, ensuring that rejected applicants receive clear, justifiable explanations for the denial.
        Arguments: self: The instance of the class. income: Annual gross income (USD). credit_score: FICO score (300-850). dti: Debt-to-income ratio (0-1.5). tenure: Employment tenure (months).
        Output: dict: A dictionary containing 'default_probability' (float), 'decision' (string: "APPROVED", "REVIEW_REQUIRED", or "DENIED"), and 'adverse_action_reasons' (list of strings or None).
        """

        prob = self.predict(income, credit_score, dti, tenure)

        # Decision thresholds
        if prob < 0.10:
            decision = "APPROVED"
            reasons = None
        elif prob < 0.20:
            decision = "REVIEW_REQUIRED"
            reasons = None
        else:
            decision = "DENIED"
            reasons = self._generate_adverse_action_reasons(
                income, credit_score, dti, tenure, prob
            )

        return {
            'default_probability': prob,
            'decision': decision,
            'adverse_action_reasons': reasons
        }
# --- End of definition_8e577cdadc704e759ca0442929dff00a block ---

# Assuming 'ConsumerCreditScoringModel' is imported from 'definition_8e577cdadc704e759ca0442929dff00a'
# For this test, the mock class above is directly used.

@pytest.fixture
def mock_model():
    """Provides an instance of the mock ConsumerCreditScoringModel."""
    return ConsumerCreditScoringModel()

def test_predict_with_decision_approved_scenario(mock_model):
    """
    Test case for an 'APPROVED' decision with low default probability.
    Corresponds to: P(default) < 0.10
    """
    income, credit_score, dti, tenure = 150000, 800, 0.15, 72 # Inputs for low risk
    result = mock_model.predict_with_decision(income, credit_score, dti, tenure)

    assert result['decision'] == "APPROVED"
    assert result['default_probability'] < 0.10
    assert result['adverse_action_reasons'] is None

def test_predict_with_decision_denied_scenario(mock_model):
    """
    Test case for a 'DENIED' decision with high default probability
    and verifies adverse action reasons (PROP-009).
    Corresponds to: P(default) >= 0.20
    """
    income, credit_score, dti, tenure = 25000, 500, 0.7, 12 # Inputs for high risk
    result = mock_model.predict_with_decision(income, credit_score, dti, tenure)

    assert result['decision'] == "DENIED"
    assert result['default_probability'] >= 0.20
    assert result['adverse_action_reasons'] is not None
    assert isinstance(result['adverse_action_reasons'], list)
    assert len(result['adverse_action_reasons']) >= 1
    # Check for specificity (reasons are not generic and have some length)
    assert all(len(reason) > 10 for reason in result['adverse_action_reasons'])
    assert "Credit score 500 below recommended minimum (640)" in result['adverse_action_reasons']
    assert "Income $25,000 insufficient for loan amount" in result['adverse_action_reasons']
    assert "Debt-to-income ratio 70.0% exceeds guideline (43%)" in result['adverse_action_reasons']
    assert "Employment history 12 months below minimum (12)" in result['adverse_action_reasons']

def test_predict_with_decision_review_required_scenario(mock_model):
    """
    Test case for a 'REVIEW_REQUIRED' decision with medium default probability.
    Corresponds to: 0.10 <= P(default) < 0.20
    """
    income, credit_score, dti, tenure = 70000, 680, 0.3, 48 # Inputs for medium risk
    result = mock_model.predict_with_decision(income, credit_score, dti, tenure)

    assert result['decision'] == "REVIEW_REQUIRED"
    assert 0.10 <= result['default_probability'] < 0.20
    assert result['adverse_action_reasons'] is None

def test_predict_with_decision_invalid_income_input(mock_model):
    """
    Test case for invalid income input (<= 0), expecting a ValueError (PROP-005).
    """
    income, credit_score, dti, tenure = 0, 700, 0.3, 24 # Invalid income
    with pytest.raises(ValueError, match="income must be positive"):
        mock_model.predict_with_decision(income, credit_score, dti, tenure)

def test_predict_with_decision_invalid_credit_score_input(mock_model):
    """
    Test case for invalid credit score input (outside 300-850), expecting a ValueError (PROP-005).
    """
    income, credit_score, dti, tenure = 50000, 900, 0.3, 24 # Invalid credit_score
    with pytest.raises(ValueError, match="credit_score must be 300-850"):
        mock_model.predict_with_decision(income, credit_score, dti, tenure)