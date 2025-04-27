import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report
from joblib import dump
from sqlalchemy import create_engine

# Database connection details
host = 'localhost'
database_name = 'courant'
username = 'root'
password = ''

# Create the database engine
engine = create_engine(f'mysql+mysqlconnector://{username}:{password}@{host}/{database_name}')

# Fetch data from the database
query = "SELECT voltage, current, power, energy, frequency, power_factor FROM readings"
df = pd.read_sql(query, engine)

# Handle missing values
df = df.fillna(df.mean())

# Prepare data for training
X = df[['voltage', 'current', 'power', 'energy', 'frequency', 'power_factor']]

# Split data into training and testing sets
X_train, X_test = train_test_split(X, test_size=0.2, random_state=42)

# Train the Isolation Forest model
model = IsolationForest(contamination='auto', random_state=42)
model.fit(X_train)

# Predict anomalies on the test set
y_pred = model.predict(X_test)

# Convert predictions to binary labels (1 for inliers, 0 for outliers)
# In Isolation Forest, 1 represents normal data and -1 represents anomalies
# Converting to 1 for normal, 0 for anomalies for the classification report
y_true = [1] * len(X_test)  # Assuming all test samples are normal for evaluation
y_pred_binary = [1 if x == 1 else 0 for x in y_pred]

# Print prediction statistics
anomaly_count = sum(1 for x in y_pred if x == -1)
normal_count = sum(1 for x in y_pred if x == 1)
print(f"Number of predicted anomalies: {anomaly_count}")
print(f"Number of predicted normal points: {normal_count}")
print(f"Anomaly percentage: {anomaly_count / len(y_pred) * 100:.2f}%")

# Evaluate the model
print("\nClassification Report:")
print(classification_report(y_true, y_pred_binary))

# Save the trained model
dump(model, 'trained_model.joblib')

print("Trained model saved to trained_model.joblib")