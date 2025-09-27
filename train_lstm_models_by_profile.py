import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from joblib import dump, load
from sqlalchemy import create_engine
from datetime import datetime, timedelta
import os

# --- Configuration ---
DB_HOST = 'localhost'
DB_NAME = 'courant'
DB_USER = 'user'
DB_PASS = 'dFSYzQ1OYBtq3qtEWBfY'
SEQUENCE_LENGTH = 10  # Number of time steps to look back
EPOCHS = 50
BATCH_SIZE = 32
INCREMENTAL_EPOCHS = 10  # Fewer epochs for incremental updates
DAYS_TO_RETRAIN = 30  # Only use last 30 days for retraining
MIN_SEQUENCES_FOR_TRAINING = 50  # Minimum sequences needed
OUTPUT_FILE = 'trained_lstm_models.joblib'
FEATURE_COLUMNS = ['voltage', 'current', 'power', 'frequency', 'power_factor']

# --- Database Connection ---
try:
    engine = create_engine(f'mysql+mysqlconnector://{DB_USER}:{DB_PASS}@{DB_HOST}/{DB_NAME}')
    print("Database connection successful.")
except Exception as e:
    print(f"Error connecting to database: {e}")
    exit()

# --- Fetch Recent Data ---
def get_recent_data_with_timestamps(engine, days_back=30):
    """Fetch only recent data with timestamps"""
    cutoff_date = datetime.now() - timedelta(days=days_back)
    
    query = """
    SELECT r.voltage, r.current, r.power, r.frequency, r.power_factor, r.profile_id, 
           r.created_at, p.alert_sensitivity
    FROM readings r
    JOIN profiles p ON r.profile_id = p.id -- Join with profiles table
    WHERE r.created_at >= %s
    ORDER BY r.profile_id, r.created_at ASC -- Order by time for sequence generation
    """
    
    df = pd.read_sql(query, engine, params=(cutoff_date,))
    df['created_at'] = pd.to_datetime(df['created_at'])
    return df

# --- Helper Function to Create Sequences ---
def create_sequences(data, seq_length):
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        x = data[i:(i + seq_length)]
        y = data[i + seq_length] # Predict the next step
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

def load_existing_models(filepath):
    """Load existing models if they exist"""
    if os.path.exists(filepath):
        try:
            return load(filepath)
        except Exception as e:
            print(f"Error loading existing models: {e}")
            return {}
    return {}

def incremental_train_profile(profile_id, profile_data, existing_model_data=None):
    """Train or update model for a specific profile"""
    
    print(f"\n--- Processing Profile ID: {profile_id} ---")
    
    # Get all data for the profile first
    profile_data_full = profile_data.copy()
    if profile_data_full.empty:
        print(f"  Skipping profile {profile_id}: No data found.")
        return None

    # Extract sensitivity (assuming it's consistent for the profile)
    alert_sensitivity = profile_data_full['alert_sensitivity'].iloc[0]
    print(f"  Alert Sensitivity: {alert_sensitivity}")

    # Select only feature columns for training
    profile_df = profile_data_full[FEATURE_COLUMNS].copy()
    print(f"  Number of readings: {len(profile_df)}")

    if len(profile_df) < SEQUENCE_LENGTH + MIN_SEQUENCES_FOR_TRAINING:
        print(f"  Skipping profile {profile_id}: Not enough data points ({len(profile_df)}) for sequence length {SEQUENCE_LENGTH}.")
        return None

    # Handle missing values (forward fill then backward fill)
    profile_df.fillna(method='ffill', inplace=True)
    profile_df.fillna(method='bfill', inplace=True) # Handle NaNs at the beginning

    # Check if NaNs remain after filling
    if profile_df.isnull().values.any():
         print(f"  Warning: Profile {profile_id} still contains NaN values after fill. Filling with 0.")
         profile_df.fillna(0, inplace=True)

    # Scaling
    if existing_model_data and 'scaler' in existing_model_data:
        # Use existing scaler but update it with new data
        scaler = existing_model_data['scaler']
        # MinMaxScaler doesn't have partial_fit, so we refit with all data
        scaler.fit(profile_df)
        print(f"  Updated existing scaler")
    else:
        scaler = MinMaxScaler()
        scaler.fit(profile_df)
        print(f"  Created new scaler")
    
    scaled_data = scaler.transform(profile_df)

    # Create sequences
    X, y = create_sequences(scaled_data, SEQUENCE_LENGTH)

    if len(X) < MIN_SEQUENCES_FOR_TRAINING:
        print(f"  Skipping profile {profile_id}: Not enough sequences after processing ({len(X)} < {MIN_SEQUENCES_FOR_TRAINING}).")
        return None

    # Split data temporally (no shuffling for time series)
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    print(f"  Training sequences: {len(X_train)}, Testing sequences: {len(X_test)}")

    # Model handling
    if existing_model_data and 'model' in existing_model_data:
        # Load existing model and continue training
        model = existing_model_data['model']
        epochs = INCREMENTAL_EPOCHS
        print(f"  Updating existing model for profile {profile_id}")
    else:
        # Create new model
        model = Sequential([
            LSTM(50, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=True),
            LSTM(50, activation='relu'),
            Dense(y_train.shape[1]) # Output layer size matches number of features
        ])
        model.compile(optimizer='adam', loss='mse') # Mean Squared Error for regression task
        epochs = EPOCHS
        print(f"  Creating new model for profile {profile_id}")

    # Train
    print(f"  Starting model training with {epochs} epochs...")
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=BATCH_SIZE,
        validation_data=(X_test, y_test),
        verbose=0,  # Reduced verbosity for daily runs
        shuffle=False
    )
    print("  Model training complete.")

    # Store the trained model, scaler, AND sensitivity
    return {
        'model': model,
        'scaler': scaler,
        'sensitivity': alert_sensitivity,
        'last_updated': datetime.now(),
        'training_samples': len(X_train),
        'sequence_length': SEQUENCE_LENGTH,
        'feature_columns': FEATURE_COLUMNS
    }

# --- Main Execution ---
if __name__ == "__main__":
    # Load existing models
    existing_models = load_existing_models(OUTPUT_FILE)
    print(f"Loaded {len(existing_models)} existing models")
    
    # Fetch recent data
    print(f"Fetching data from last {DAYS_TO_RETRAIN} days...")
    df = get_recent_data_with_timestamps(engine, DAYS_TO_RETRAIN)
    print(f"Successfully fetched {len(df)} readings.")
    
    if df.empty:
        print("No recent data found. Exiting.")
        exit()

    # Train Models Per Profile
    trained_objects = {} # Will store {'profile_id': {'model': model, 'scaler': scaler, 'sensitivity': alert_sensitivity}}
    profile_ids = df['profile_id'].unique()
    print(f"Found {len(profile_ids)} unique profiles. Starting training...")

    for profile_id in profile_ids:
        profile_data = df[df['profile_id'] == profile_id].copy()
        profile_data = profile_data.sort_values('created_at')  # Ensure temporal order
        
        existing_data = existing_models.get(profile_id)
        result = incremental_train_profile(profile_id, profile_data, existing_data)
        
        if result:
            trained_objects[profile_id] = result
            print(f"  ✓ Model, scaler, and sensitivity stored for profile {profile_id}.")
        elif existing_data:
            # Keep existing model if update fails
            trained_objects[profile_id] = existing_data
            print(f"  ⚠ Kept existing model for profile {profile_id}")
        else:
            print(f"  ✗ Failed to train model for profile {profile_id}")

    # Save Trained Models and Scalers
    if trained_objects:
        try:
            dump(trained_objects, OUTPUT_FILE)
            print(f"\n✓ All LSTM models saved to {OUTPUT_FILE}")
            print(f"Total profiles: {len(trained_objects)}")
            
            # Print summary
            new_models = sum(1 for p_id in trained_objects if p_id not in existing_models)
            updated_existing = len(trained_objects) - new_models
            print(f"New models: {new_models}")
            print(f"Updated models: {updated_existing}")
            
        except Exception as e:
            print(f"✗ Error saving models: {e}")
    else:
        print("\n⚠ No models were trained or saved.")

    print("Script finished.")
