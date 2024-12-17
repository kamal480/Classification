from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import os

# Load data
data = pd.read_csv('src/models/classification_data.csv')
X = data.drop(columns=['class'])
y = data['class']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Save the trained model
output_dir = 'src/models/'
os.makedirs(output_dir, exist_ok=True)
joblib.dump(model, os.path.join(output_dir, 'classification_model.pkl'))
print("Model saved successfully in 'src/models/classification_model.pkl'")

import joblib
joblib.dump(model, 'src/models/classification_model.pkl')
