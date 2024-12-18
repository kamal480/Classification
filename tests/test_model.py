import pytest
import joblib
import pandas as pd

def test_model_prediction():
    # Load model
    model = joblib.load('src/models/classification_model.pkl')

    # Load data
    data = pd.read_csv('data/classification_data.csv')
    X = data.drop(columns=['class'])

    # Make predictions
    predictions = model.predict(X)
    print(predictions)
    # Assertions
    assert len(predictions) == len(X)
    assert set(predictions) <= {0, 1}
