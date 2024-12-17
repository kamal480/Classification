from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pandas as pd

# Load data
data = pd.read_csv('classification_data.csv')
X = data.drop(columns=['class'])
y = data['class']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Save the trained model
import joblib
joblib.dump(model, 'classification_model.pkl')
