import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
# shap is implicitly expected for the explainer attribute, but not directly imported here for init tests

# definition_7878e7bd381247faa8d001dd6c5c4907 block
from definition_7878e7bd381247faa8d001dd6c5c4907 import ConsumerCreditScoringModel
# definition_7878e7bd381247faa8d001dd6c5c4907 block

def test_consumer_credit_scoring_model_init_component_types():
    """
    Test that the ConsumerCreditScoringModel's internal components (model, scaler)
    are initialized to the correct types upon instantiation.
    """
    model_instance = ConsumerCreditScoringModel()

    assert isinstance(model_instance.model, LogisticRegression), \
        "self.model should be an instance of LogisticRegression"
    assert isinstance(model_instance.scaler, StandardScaler), \
        "self.scaler should be an instance of StandardScaler"

def test_consumer_credit_scoring_model_init_logistic_regression_parameters():
    """
    Test that the LogisticRegression model within ConsumerCreditScoringModel
    is initialized with the specified parameters according to the technical specification.
    """
    model_instance = ConsumerCreditScoringModel()

    assert model_instance.model.penalty == 'l2', \
        "LogisticRegression penalty should be 'l2'"
    assert model_instance.model.C == 1 / 0.01, \
        "LogisticRegression C (inverse of regularization strength) should be 1/0.01"
    assert model_instance.model.solver == 'lbfgs', \
        "LogisticRegression solver should be 'lbfgs'"
    assert model_instance.model.max_iter == 1000, \
        "LogisticRegression max_iter should be 1000"
    assert model_instance.model.class_weight == 'balanced', \
        "LogisticRegression class_weight should be 'balanced'"

def test_consumer_credit_scoring_model_init_explainer_initial_state():
    """
    Test that the SHAP explainer attribute is initially set to None, as per the design.
    """
    model_instance = ConsumerCreditScoringModel()

    assert model_instance.explainer is None, \
        "self.explainer should be initialized to None"