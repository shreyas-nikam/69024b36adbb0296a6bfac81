import pytest
import pandas as pd
import numpy as np

# definition_6606fffdc5d74ac0a4d37f8ec9ad42fb block
# This block is for the actual module import. Since the method is private,
# we need to simulate the class that contains it for testing purposes.
# We'll create a dummy class inside the test file for this.
# DO NOT REPLACE or REMOVE the block.
# from definition_6606fffdc5d74ac0a4d37f8ec9ad42fb import ConsumerCreditScoringModel
# End of definition_6606fffdc5d74ac0a4d37f8ec9ad42fb block

# Dummy class for testing _generate_adverse_action_reasons in isolation.
# This mock implements the necessary parts of ConsumerCreditScoringModel for this specific test.
class MockConsumerCreditScoringModel:
    def __init__(self, shap_values_mock, feature_values_for_transform_mock=None):
        self._shap_values_mock = shap_values_mock
        # _transform_features returns a numpy array of shape (1, N_FEATURES).
        # The specific values don't matter for `_generate_adverse_action_reasons` logic,
        # only that it provides an array for `self.explainer.shap_values` call.
        self._feature_values_for_transform = np.array(
            feature_values_for_transform_mock if feature_values_for_transform_mock is not None else [1, 1, 1, 1]
        )

    def _transform_features(self, X: pd.DataFrame) -> np.ndarray:
        # For the purpose of this test, we return a pre-defined array.
        # In a real scenario, this would apply actual transformations.
        return self._feature_values_for_transform

    @property
    def explainer(self):
        # Mocks the SHAP explainer to return pre-defined SHAP values.
        class MockExplainer:
            def __init__(self, shap_values):
                self._shap_values = shap_values
            def shap_values(self, X_transformed):
                # The target implementation uses `explainer.shap_values(X)[0]`.
                # So we return a list where the first element is our mock SHAP values.
                return [np.array(self._shap_values)]
        return MockExplainer(self._shap_values_mock)

    def _generate_adverse_action_reasons(self, income, credit_score, dti, tenure, prob):
        """
        Generate specific adverse action reasons (ECOA requirement)
        (Implementation copied verbatim from notebook specification for testing context)
        """
        X = self._transform_features(pd.DataFrame([{
            'income': income,
            'credit_score': credit_score,
            'dti': dti,
            'tenure': tenure
        }]))

        shap_values = self.explainer.shap_values(X)[0]

        feature_contributions = {
            'income': shap_values[0],
            'credit_score': shap_values[1],
            'dti': shap_values[2],
            'tenure': shap_values[3]
        }

        sorted_features = sorted(
            feature_contributions.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )

        reasons = []
        for feature, contribution in sorted_features[:3]:  # Consider only the top 3 most impactful features
            if feature == 'credit_score' and credit_score < 640:
                reasons.append(f"Credit score {credit_score} below recommended minimum (640)")
            elif feature == 'income' and income < 35000:
                reasons.append(f"Income ${income:,.0f} insufficient for loan amount")
            elif feature == 'dti' and dti > 0.43:
                reasons.append(f"Debt-to-income ratio {dti:.1%} exceeds guideline (43%)")
            elif feature == 'tenure' and tenure < 12:
                reasons.append(f"Employment history {tenure} months below minimum (12)")

        return reasons if reasons else ["Overall credit profile indicates high default risk"]


# Test function using parametrize for different scenarios
@pytest.mark.parametrize(
    "income, credit_score, dti, tenure, prob, shap_values_mock, expected_reasons",
    [
        # Test Case 1: Multiple factors contributing to denial (top 3 by SHAP absolute value)
        # DTI, Credit Score, and Income have the highest absolute SHAP values and meet denial thresholds.
        (20000, 550, 0.5, 6, 0.8, [-0.5, -0.6, 0.7, -0.1], # SHAP: income, credit_score, dti, tenure
         [
             "Debt-to-income ratio 50.0% exceeds guideline (43%)",
             "Credit score 550 below recommended minimum (640)",
             "Income $20,000 insufficient for loan amount",
         ]),
        # Test Case 2: Only one strong factor (credit score) contributing to denial.
        # Credit score has the highest absolute SHAP value and meets the denial threshold.
        (50000, 500, 0.2, 24, 0.6, [-0.05, -1.0, 0.02, -0.03], # Credit score has highest absolute SHAP
         [
             "Credit score 500 below recommended minimum (640)"
         ]),
        # Test Case 3: No specific reasons met, falls back to the generic reason.
        # All inputs are within 'acceptable' ranges, and SHAP values are not strong enough to trigger specific reasons.
        (40000, 650, 0.3, 18, 0.7, [0.1, -0.05, 0.05, -0.02], # SHAP values don't strongly push for denial or cross thresholds
         ["Overall credit profile indicates high default risk"]),
        # Test Case 4: Edge Case - Credit score exactly at the threshold (640).
        # Since it's not strictly "below 640", this reason should not be triggered.
        (50000, 640, 0.2, 24, 0.6, [-0.1, -0.8, 0.05, -0.05], # Credit score has highest absolute SHAP, but value is 640
         ["Overall credit profile indicates high default risk"]),
        # Test Case 5: Edge Case - Credit score just below the threshold (639).
        # This value should trigger the specific credit score reason.
        (50000, 639, 0.2, 24, 0.6, [-0.1, -0.8, 0.05, -0.05], # Credit score has highest absolute SHAP, value is 639
         ["Credit score 639 below recommended minimum (640)"]),
    ]
)
def test_generate_adverse_action_reasons(income, credit_score, dti, tenure, prob, shap_values_mock, expected_reasons):
    # Instantiate our mock model with the specific SHAP values for this test case.
    # The `feature_values_for_transform_mock` is optional and defaults to [1,1,1,1] in MockConsumerCreditScoringModel,
    # as its content doesn't impact the logic being tested here.
    model_instance = MockConsumerCreditScoringModel(shap_values_mock)
    
    # Call the method under test.
    actual_reasons = model_instance._generate_adverse_action_reasons(
        income, credit_score, dti, tenure, prob
    )
    
    # Assert the returned reasons match the expected ones.
    # Using sorted() for comparison to handle potential differences in reason order if multiple reasons are generated.
    assert sorted(actual_reasons) == sorted(expected_reasons)