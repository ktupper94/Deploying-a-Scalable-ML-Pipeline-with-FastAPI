import pytest
from sklearn.ensemble import RandomForestClassifier
from train_model import X_train, y_train
from ml.model import compute_model_metrics, train_model
    
    

def test_compute_model_metrics():
    """
    Tests to ensure the compute_model_metrics function returns expected types
    """
    
    y_true = [0, 1, 1, 0, 1]
    y_pred = [0, 1, 0, 0, 1]
    
    
    precision, recall, f1 = compute_model_metrics(y_true, y_pred)
    
    
    assert isinstance(precision, float)
    assert isinstance(recall, float)
    assert isinstance(f1, float)
    


def test_model_trains_successfully():
    """
    Tests the train_model function to ensure it trains a model successfully
    """
    model = train_model(X_train, y_train)
    assert hasattr(model, 'n_features_in_'), "Model has not been trained (fitted)."

def test_model_uses_expected_algorithm():
    """
    Tests the train_model function to ensure it uses the proper algorithm
    """
    model = train_model(X_train, y_train)
    
    assert isinstance(model, RandomForestClassifier), "Expected model to be a RandomForestClassifier"
