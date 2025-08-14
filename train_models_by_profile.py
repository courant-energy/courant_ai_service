import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from joblib import dump, load
from sqlalchemy import create_engine
from datetime import datetime, timedelta
import os

# Database connection details
host = 'localhost'
database_name = 'courant'
username = 'user'
password = 'dFSYzQ1OYBtq3qtEWBfY'

# Configuration
DAYS_TO_RETRAIN = 30  # Only use last 30 days for retraining
TEMPORAL_WINDOW = 5   # Number of recent readings for temporal features
MIN_SAMPLES = 100     # Minimum samples needed for training
FEATURE_COLUMNS = ['voltage', 'current', 'power', 'frequency', 'power_factor']
OUTPUT_FILE = 'trained_models.joblib'

def get_recent_data_with_timestamps(engine, days_back=30):
    """Fetch only recent data with timestamps"""
    cutoff_date = datetime.now() - timedelta(days=days_back)
    
    query = """
    SELECT r.voltage, r.current, r.power, r.frequency, r.power_factor, 
           r.profile_id, r.created_at, p.alert_sensitivity
    FROM readings r
    JOIN profiles p ON r.profile_id = p.id
    WHERE r.created_at >= %s
    ORDER BY r.profile_id, r.created_at ASC
    """
    
    df = pd.read_sql(query, engine, params=[cutoff_date])
    df['created_at'] = pd.to_datetime(df['created_at'])
    return df

def create_temporal_features(df, window_size=5):
    """Create features that capture temporal patterns"""
    if len(df) < window_size + 1:
        return None
    
    temporal_features = []
    
    for i in range(window_size, len(df)):
        # Current reading
        current = df.iloc[i][FEATURE_COLUMNS].values
        
        # Recent history window
        history = df.iloc[i-window_size:i][FEATURE_COLUMNS].values
        
        # Statistical features from history
        history_mean = np.mean(history, axis=0)
        history_std = np.std(history, axis=0)
        history_max = np.max(history, axis=0)
        history_min = np.min(history, axis=0)
        
        # Trend features (difference from recent average)
        trend = current - history_mean
        
        # Volatility features (normalized deviation)
        volatility = np.abs(trend) / (history_std + 1e-8)
        
        # Rate of change features
        if i > window_size:
            prev_reading = df.iloc[i-1][FEATURE_COLUMNS].values
            rate_of_change = current - prev_reading
        else:
            rate_of_change = np.zeros_like(current)
        
        # Time-based features
        timestamp = df.iloc[i]['created_at']
        hour_of_day = timestamp.hour
        day_of_week = timestamp.weekday()
        is_weekend = 1 if day_of_week >= 5 else 0
        
        # Combine all features
        feature_vector = np.concatenate([
            current,           # 5 features: current readings
            history_mean,      # 5 features: recent averages
            history_std,       # 5 features: recent volatility
            trend,             # 5 features: trend from average
            volatility,        # 5 features: normalized volatility
            rate_of_change,    # 5 features: rate of change
            [hour_of_day, day_of_week, is_weekend]  # 3 features: time context
        ])
        
        temporal_features.append(feature_vector)
    
    return np.array(temporal_features)

def load_existing_models(filepath):
    """Load existing models if they exist"""
    if os.path.exists(filepath):
        try:
            return load(filepath)
        except Exception as e:
            print(f"Error loading existing models: {e}")
            return {}
    return {}

def train_model_for_profile(profile_id, profile_data, existing_model_data=None):
    """Train or update Isolation Forest for a specific profile"""
    
    print(f"\nProcessing Profile ID: {profile_id}")
    print(f"  Number of readings: {len(profile_data)}")
    
    # Get alert sensitivity
    alert_sensitivity = profile_data['alert_sensitivity'].iloc[0]
    
    # Map alert sensitivity to contamination
    contamination_map = {
        'low': 0.05,
        'medium': 0.10,
        'high': 0.15
    }
    contamination = contamination_map.get(alert_sensitivity, 0.10)
    print(f"  Alert sensitivity: {alert_sensitivity}, Contamination: {contamination}")
    
    # Handle missing values
    profile_clean = profile_data.copy()
    for col in FEATURE_COLUMNS:
        profile_clean[col].fillna(method='ffill', inplace=True)
        profile_clean[col].fillna(method='bfill', inplace=True)
        profile_clean[col].fillna(profile_clean[col].mean(), inplace=True)
    
    # Create temporal features
    temporal_features = create_temporal_features(profile_clean, TEMPORAL_WINDOW)
    
    if temporal_features is None or len(temporal_features) < MIN_SAMPLES:
        print(f"  Insufficient data for profile {profile_id} (need {MIN_SAMPLES}, have {len(temporal_features) if temporal_features is not None else 0})")
        return None
    
    print(f"  Created {len(temporal_features)} temporal feature vectors")
    print(f"  Feature vector size: {temporal_features.shape[1]}")
    
    # Scaling
    if existing_model_data and 'scaler' in existing_model_data:
        scaler = existing_model_data['scaler']
        # Refit scaler with new data
        scaler.fit(temporal_features)
        print(f"  Updated existing scaler")
    else:
        scaler = StandardScaler()
        scaler.fit(temporal_features)
        print(f"  Created new scaler")
    
    scaled_features = scaler.transform(temporal_features)
    
    # Split data temporally (not randomly)
    split_idx = int(len(scaled_features) * 0.8)
    X_train = scaled_features[:split_idx]
    X_test = scaled_features[split_idx:]
    
    print(f"  Training samples: {len(X_train)}, Testing samples: {len(X_test)}")
    
    # Train Isolation Forest
    model = IsolationForest(
        n_estimators=300,
        contamination=contamination,
        random_state=42,
        n_jobs=-1  # Use all CPU cores
    )
    model.fit(X_train)
    print(f"  Model trained with contamination={contamination}")
    
    # Evaluate on test set
    if len(X_test) > 0:
        y_pred = model.predict(X_test)
        anomaly_count = sum(1 for x in y_pred if x == -1)
        normal_count = sum(1 for x in y_pred if x == 1)
        anomaly_percentage = anomaly_count / len(y_pred) * 100
        
        print(f"  Test set predictions:")
        print(f"    Anomalies: {anomaly_count} ({anomaly_percentage:.2f}%)")
        print(f"    Normal: {normal_count} ({100-anomaly_percentage:.2f}%)")
    
    return {
        'model': model,
        'scaler': scaler,
        'sensitivity': alert_sensitivity,
        'contamination': contamination,
        'temporal_window': TEMPORAL_WINDOW,
        'feature_count': temporal_features.shape[1],
        'training_samples': len(X_train),
        'last_updated': datetime.now(),
        'feature_names': (
            [f'{col}_current' for col in FEATURE_COLUMNS] +
            [f'{col}_mean' for col in FEATURE_COLUMNS] +
            [f'{col}_std' for col in FEATURE_COLUMNS] +
            [f'{col}_trend' for col in FEATURE_COLUMNS] +
            [f'{col}_volatility' for col in FEATURE_COLUMNS] +
            [f'{col}_rate_change' for col in FEATURE_COLUMNS] +
            ['hour_of_day', 'day_of_week', 'is_weekend']
        )
    }

# Create database engine
engine = create_engine(f'mysql+mysqlconnector://{username}:{password}@{host}/{database_name}')

# Load existing models
existing_models = load_existing_models(OUTPUT_FILE)
print(f"Loaded {len(existing_models)} existing models")

# Get recent data with timestamps
print(f"Fetching data from last {DAYS_TO_RETRAIN} days...")
df = get_recent_data_with_timestamps(engine, DAYS_TO_RETRAIN)
print(f"Fetched {len(df)} readings")

if df.empty:
    print("No recent data found. Exiting.")
    exit()

# Train models per profile
updated_models = {}
profile_ids = df['profile_id'].unique()
print(f"Found {len(profile_ids)} unique profiles")

for profile_id in profile_ids:
    profile_data = df[df['profile_id'] == profile_id].copy()
    profile_data = profile_data.sort_values('created_at')  # Ensure temporal order
    
    existing_data = existing_models.get(profile_id)
    result = train_model_for_profile(profile_id, profile_data, existing_data)
    
    if result:
        updated_models[profile_id] = result
        print(f"  ✓ Successfully trained model for profile {profile_id}")
    elif existing_data:
        # Keep existing model if new training fails
        updated_models[profile_id] = existing_data
        print(f"  ⚠ Kept existing model for profile {profile_id}")
    else:
        print(f"  ✗ Failed to train model for profile {profile_id}")

# Save updated models
if updated_models:
    try:
        dump(updated_models, OUTPUT_FILE)
        print(f"\n✓ All models saved to {OUTPUT_FILE}")
        print(f"Total profiles: {len(updated_models)}")
        
        # Print summary
        new_models = sum(1 for p_id in updated_models if p_id not in existing_models)
        updated_existing = len(updated_models) - new_models
        print(f"New models: {new_models}")
        print(f"Updated models: {updated_existing}")
        
    except Exception as e:
        print(f"✗ Error saving models: {e}")
else:
    print("\n⚠ No models to save")

print("\nTraining completed.")
