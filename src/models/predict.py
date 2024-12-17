import pandas as pd
import joblib

# Load data
data = pd.read_csv('src/models/classification_data.csv')
X = data.drop(columns=['class'])

# Load model
model = joblib.load('src/models/classification_model.pkl')

# Predictions
predictions = model.predict(X)

# Save predictions
data['predictions'] = predictions
data.to_csv('classification_predictions.csv', index=False)
