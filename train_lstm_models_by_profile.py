import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from joblib import dump
from sqlalchemy import create_engine
import os

# --- Configuration ---
DB_HOST = 'localhost'
DB_NAME = 'courant'
DB_USER = 'root'
DB_PASS = ''
SEQUENCE_LENGTH = 10  # Number of time steps to look back
EPOCHS = 50
BATCH_SIZE = 32
OUTPUT_FILE = 'trained_lstm_models.joblib'
FEATURE_COLUMNS = ['voltage', 'current', 'power', 'frequency', 'power_factor']

# --- Database Connection ---
try:
    engine = create_engine(f'mysql+mysqlconnector://{DB_USER}:{DB_PASS}@{DB_HOST}/{DB_NAME}')
    print("Database connection successful.")
except Exception as e:
    print(f"Error connecting to database: {e}")
    exit()

# --- Fetch Data ---
query = """
SELECT r.voltage, r.current, r.power, r.frequency, r.power_factor, r.profile_id, p.alert_sensitivity
FROM readings r
JOIN profiles p ON r.profile_id = p.id -- Join with profiles table
ORDER BY r.profile_id, r.created_at ASC -- Order by time for sequence generation
"""
try:
    df = pd.read_sql(query, engine)
    print(f"Successfully fetched {len(df)} readings.")
except Exception as e:
    print(f"Error fetching data: {e}")
    exit()

# --- Helper Function to Create Sequences ---
def create_sequences(data, seq_length):
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        x = data[i:(i + seq_length)]
        y = data[i + seq_length] # Predict the next step
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

# --- Train Models Per Profile ---
trained_objects = {} # Will store {'profile_id': {'model': model, 'scaler': scaler, 'sensitivity': alert_sensitivity}}
profile_ids = df['profile_id'].unique()
print(f"Found {len(profile_ids)} unique profiles. Starting training...")

for profile_id in profile_ids:
    print(f"\n--- Processing Profile ID: {profile_id} ---")
    # Get all data for the profile first
    profile_data_full = df[df['profile_id'] == profile_id].copy()
    if profile_data_full.empty:
        print(f"  Skipping profile {profile_id}: No data found.")
        continue

    # Extract sensitivity (assuming it's consistent for the profile)
    alert_sensitivity = profile_data_full['alert_sensitivity'].iloc[0]
    print(f"  Alert Sensitivity: {alert_sensitivity}")

    # Select only feature columns for training
    profile_df = profile_data_full[FEATURE_COLUMNS].copy()
    print(f"  Number of readings: {len(profile_df)}")

    if len(profile_df) < SEQUENCE_LENGTH + 1:
        print(f"  Skipping profile {profile_id}: Not enough data points ({len(profile_df)}) for sequence length {SEQUENCE_LENGTH}.")
        continue

    # Handle missing values (simple forward fill, consider other methods if needed)
    profile_df.fillna(method='ffill', inplace=True)
    profile_df.fillna(method='bfill', inplace=True) # Handle NaNs at the beginning

    # Check if NaNs remain after filling
    if profile_df.isnull().values.any():
         print(f"  Warning: Profile {profile_id} still contains NaN values after fill. Filling with 0.")
         profile_df.fillna(0, inplace=True)


    # Scale features
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(profile_df)

    # Create sequences
    X, y = create_sequences(scaled_data, SEQUENCE_LENGTH)

    if len(X) == 0:
        print(f"  Skipping profile {profile_id}: Not enough data to create sequences after processing.")
        continue

    # Split data (no shuffling for time series)
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    print(f"  Training sequences: {len(X_train)}, Testing sequences: {len(X_test)}")

    # --- Build LSTM Model ---
    model = Sequential([
        LSTM(50, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=True),
        LSTM(50, activation='relu'),
        Dense(y_train.shape[1]) # Output layer size matches number of features
    ])
    model.compile(optimizer='adam', loss='mse') # Mean Squared Error for regression task
    print("  Model compiled.")
    # model.summary() # Optional: Print model summary

    # --- Train Model ---
    print("  Starting model training...")
    history = model.fit(
        X_train, y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(X_test, y_test),
        verbose=1, # Set to 0 for less output, 1 for progress bar
        shuffle=False # Important for time series
    )
    print("  Model training complete.")

    # Store the trained model, scaler, AND sensitivity
    trained_objects[profile_id] = {
        'model': model,
        'scaler': scaler,
        'sensitivity': alert_sensitivity # Add sensitivity here
    }
    print(f"  Model, scaler, and sensitivity stored for profile {profile_id}.")

# --- Save Trained Models and Scalers ---
if trained_objects:
    try:
        dump(trained_objects, OUTPUT_FILE)
        print(f"\nAll trained LSTM models and scalers saved to {OUTPUT_FILE}")
        print(f"Total number of profiles processed and saved: {len(trained_objects)}")
    except Exception as e:
        print(f"\nError saving models to {OUTPUT_FILE}: {e}")
else:
    print("\nNo models were trained or saved.")

print("Script finished.")
