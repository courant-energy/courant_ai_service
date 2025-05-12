import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report
from joblib import dump
from sqlalchemy import create_engine

# Database connection details
host = 'localhost'
database_name = 'courant'
username = 'user'
password = 'dFSYzQ1OYBtq3qtEWBfY'

# Create the database engine
engine = create_engine(f'mysql+mysqlconnector://{username}:{password}@{host}/{database_name}')

# Fetch data from the database
query = """
SELECT r.voltage, r.current, r.power, r.frequency, r.power_factor, r.profile_id, p.alert_sensitivity
FROM readings r
JOIN profiles p ON r.profile_id = p.id
"""
df = pd.read_sql(query, engine)

# Preprocess data and train models per profile
models = {}
profile_ids = df['profile_id'].unique()
print(f"Found {len(profile_ids)} unique profiles")

for profile_id in profile_ids:
    print(f"\nProcessing Profile ID: {profile_id}")
    profile_df = df[df['profile_id'] == profile_id].copy()
    print(f"  Number of readings: {len(profile_df)}")

    # Get the alert sensitivity for this profile
    alert_sensitivity = profile_df['alert_sensitivity'].iloc[0]
    
    # Map alert sensitivity to contamination value
    contamination = 0.10  # Default to medium
    if alert_sensitivity == 'low':
        contamination = 0.05
    elif alert_sensitivity == 'medium':
        contamination = 0.10
    elif alert_sensitivity == 'high':
        contamination = 0.15
    print(f"  Alert sensitivity: {alert_sensitivity}, Contamination: {contamination}")
    
    # Store alert_sensitivity separately and drop it before filling missing values
    profile_df = profile_df.drop(columns=['alert_sensitivity'])
    
    # Handle missing values for the current profile (numeric columns only)
    profile_df = profile_df.fillna(profile_df.mean())

    # Prepare data for training
    X = profile_df[['voltage', 'current', 'power', 'frequency', 'power_factor']]

    # Split data into training and testing sets for this profile
    X_train, X_test = train_test_split(X, test_size=0.2, random_state=42)
    print(f"  Training samples: {len(X_train)}, Testing samples: {len(X_test)}")

    # Train the Isolation Forest model using alert sensitivity to set contamination
    model = IsolationForest(contamination=contamination, random_state=42)
    model.fit(X_train)
    models[profile_id] = model
    print(f"  Model trained for profile {profile_id} with contamination={contamination}")

    # Predict anomalies on the test set
    y_pred = model.predict(X_test)
    
    # Convert predictions to binary labels
    # In Isolation Forest, 1 represents normal data and -1 represents anomalies
    # For the classification report, we convert to 1 for normal, 0 for anomalies
    y_true = [1] * len(X_test)  # Assuming all test samples are normal for evaluation
    y_pred_binary = [1 if x == 1 else 0 for x in y_pred]

    # Print prediction statistics
    anomaly_count = sum(1 for x in y_pred if x == -1)
    normal_count = sum(1 for x in y_pred if x == 1)
    print(f"  Number of predicted anomalies: {anomaly_count}")
    print(f"  Number of predicted normal points: {normal_count}")
    print(f"  Anomaly percentage: {anomaly_count / len(y_pred) * 100:.2f}%")

    # Evaluate the model
    print(f"\n  Classification Report for Profile {profile_id}:")
    print(classification_report(y_true, y_pred_binary))

# Save the trained models
dump(models, 'trained_models.joblib')

print("\nAll trained models saved to trained_models.joblib")
print(f"Total number of models saved: {len(models)}")
