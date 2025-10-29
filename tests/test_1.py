import pytest
import pandas as pd
import numpy as np
import shap # Used internally by the model

# Placeholder for your module import
from definition_d1332ccae640453dae73d0304d1db0c4 import ConsumerCreditScoringModel

@pytest.fixture
def sample_data():
    """Provides valid sample data for X_train and y_train."""
    X = pd.DataFrame({
        'income': [50000, 70000, 30000, 100000, 45000],
        'credit_score': [700, 750, 600, 800, 680],
        'dti': [0.3, 0.2, 0.45, 0.15, 0.35],
        'tenure': [24, 60, 12, 120, 36]
    })
    y = pd.Series([0, 0, 1, 0, 1], dtype=int) # 0=no default, 1=default
    return X, y

@pytest.fixture
def model_instance():
    """Provides a fresh instance of ConsumerCreditScoringModel for each test."""
    return ConsumerCreditScoringModel()

# Test Case 1: Successful training with valid data
def test_train_successful(model_instance, sample_data):
    """
    Tests if the train method successfully trains the model and initializes the SHAP explainer
    with valid input data.
    """
    X_train, y_train = sample_data
    
    # The train method should return self
    trained_model = model_instance.train(X_train, y_train)

    assert trained_model is model_instance
    assert trained_model.model is not None
    assert hasattr(trained_model.model, 'coef_') and trained_model.model.coef_ is not None
    assert hasattr(trained_model.model, 'intercept_') and trained_model.model.intercept_ is not None
    assert trained_model.model.coef_.shape == (1, 4) # Expected shape for 1 target, 4 features
    assert trained_model.explainer is not None
    assert isinstance(trained_model.explainer, shap.LinearExplainer)

# Test Case 2: Training with empty data
def test_train_empty_data(model_instance):
    """
    Tests how the train method handles empty input data for X_train and y_train.
    Expects a ValueError from scikit-learn's fit method or SHAP initialization.
    """
    X_empty = pd.DataFrame(columns=['income', 'credit_score', 'dti', 'tenure'])
    y_empty = pd.Series([], dtype=int)

    with pytest.raises((ValueError, np.linalg.LinAlgError, IndexError), 
                       match="empty array|cannot convert to float array|cannot be converted to an array scalar|Found input variables with inconsistent numbers of samples|Expected 2D array, got 1D array instead"):
        model_instance.train(X_empty, y_empty)

# Test Case 3: Training with X_train missing required columns
def test_train_missing_columns(model_instance, sample_data):
    """
    Tests if the train method correctly identifies and rejects X_train dataframes
    that are missing essential feature columns.
    """
    X_train, y_train = sample_data
    X_train_missing_col = X_train.drop(columns=['dti']) # Remove a required column

    with pytest.raises(KeyError, match="'dti'"):
        model_instance.train(X_train_missing_col, y_train)

# Test Case 4: Training with mismatched lengths of X_train and y_train
def test_train_mismatched_lengths(model_instance, sample_data):
    """
    Tests if the train method handles cases where the number of samples in X_train
    does not match the number of samples in y_train.
    """
    X_train, y_train = sample_data
    y_mismatched = y_train.iloc[:-1] # Make y_train shorter than X_train

    with pytest.raises(ValueError, match="Found input variables with inconsistent numbers of samples|X and y have incompatible sizes"):
        model_instance.train(X_train, y_mismatched)

# Test Case 5: Training with X_train not a Pandas DataFrame
def test_train_invalid_X_type(model_instance, sample_data):
    """
    Tests if the train method properly handles X_train inputs that are not
    pandas DataFrames, which is required by the internal feature transformation.
    """
    _, y_train = sample_data
    X_invalid_type = np.array(sample_data[0]) # Pass a numpy array instead of a DataFrame

    with pytest.raises(AttributeError, match="('numpy.ndarray' object has no attribute 'copy'|'numpy.ndarray' object is not subscriptable)"):
        model_instance.train(X_invalid_type, y_train)
