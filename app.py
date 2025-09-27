from flask import Flask, request, jsonify
from joblib import load
import os
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime
from sqlalchemy import create_engine
import pandas as pd

app = Flask(__name__)
models = None  # Global variable to hold the loaded Isolation Forest models dictionary
lstm_models = None # Global variable to hold the loaded LSTM models dictionary

# Database configuration
DB_HOST = 'localhost'
DB_NAME = 'courant'
DB_USER = 'user'
DB_PASS = 'dFSYzQ1OYBtq3qtEWBfY'

def get_database_engine():
    """Get database engine for fetching recent data"""
    return create_engine(f'mysql+mysqlconnector://{DB_USER}:{DB_PASS}@{DB_HOST}/{DB_NAME}')

def get_recent_readings_for_temporal_features(profile_id, window_size=5):
    """Fetch recent readings to create temporal features for Isolation Forest"""
    try:
        engine = get_database_engine()
        query = """
        SELECT voltage, current, power, frequency, power_factor, created_at
        FROM readings
        WHERE profile_id = %s
        ORDER BY created_at DESC
        LIMIT %s
        """
        df = pd.read_sql(query, engine, params=(profile_id, window_size))
        return df
    except Exception as e:
        app.logger.error(f"Error fetching recent readings for profile {profile_id}: {e}")
        return pd.DataFrame()

def get_recent_sequence_for_lstm(profile_id, sequence_length=10):
    """Fetch recent readings sequence for LSTM prediction"""
    try:
        engine = get_database_engine()
        query = """
        SELECT voltage, current, power, frequency, power_factor, created_at
        FROM readings
        WHERE profile_id = %s
        ORDER BY created_at DESC
        LIMIT %s
        """
        df = pd.read_sql(query, engine, params=(profile_id, sequence_length))
        return df.iloc[::-1]  # Reverse to get chronological order
    except Exception as e:
        app.logger.error(f"Error fetching sequence for profile {profile_id}: {e}")
        return pd.DataFrame()

def create_temporal_features_for_prediction(current_data, recent_data, timestamp):
    """Create temporal features similar to training script"""
    feature_columns = ['voltage', 'current', 'power', 'frequency', 'power_factor']
    
    # Time-based features
    hour_of_day = timestamp.hour
    day_of_week = timestamp.weekday()
    is_weekend = 1 if day_of_week >= 5 else 0
    
    # Current values
    current_features = [current_data.get(col, 0) for col in feature_columns]
    
    if len(recent_data) > 0:
        # Historical statistics
        historical_means = [recent_data[col].mean() for col in feature_columns]
        historical_stds = [recent_data[col].std() for col in feature_columns]
        
        # Simple trends (current vs recent mean)
        trends = [(current_data.get(col, 0) - recent_data[col].mean()) / (recent_data[col].std() + 1e-6) 
                 for col in feature_columns]
        
        # Volatility (coefficient of variation)
        volatility = [recent_data[col].std() / (recent_data[col].mean() + 1e-6) for col in feature_columns]
        
        # Rate of change (simple)
        if len(recent_data) >= 2:
            rate_of_change = [(recent_data[col].iloc[-1] - recent_data[col].iloc[-2]) / (recent_data[col].iloc[-2] + 1e-6) 
                            for col in feature_columns]
        else:
            rate_of_change = [0] * len(feature_columns)
    else:
        # Fallback if no recent data
        historical_means = current_features.copy()
        historical_stds = [0] * len(feature_columns)
        trends = [0] * len(feature_columns)
        volatility = [0] * len(feature_columns)
        rate_of_change = [0] * len(feature_columns)
    
    # Combine all features
    feature_vector = (current_features + historical_means + historical_stds + 
                     trends + volatility + rate_of_change + 
                     [hour_of_day, day_of_week, is_weekend])
    
    return feature_vector

# For newer Flask versions (2.0+) which don't support @app.before_first_request
@app.before_request
def load_models_if_needed():
    global models, lstm_models
    # Load Isolation Forest models
    if models is None:
        try:
            models_path = os.path.join(os.path.dirname(__file__), 'trained_models.joblib')
            if os.path.exists(models_path):
                models = load(models_path)
                app.logger.info(f"Isolation Forest models loaded successfully. Available profile IDs: {list(models.keys())}")
            else:
                app.logger.warning(f"Isolation Forest model file not found at {models_path}. '/predict' endpoint will not work.")
                models = {} # Initialize as empty dict if file not found
        except Exception as e:
            app.logger.error(f"Error loading Isolation Forest models: {str(e)}")
            models = {} # Initialize as empty dict on error

    # Load LSTM models
    if lstm_models is None:
        try:
            lstm_models_path = os.path.join(os.path.dirname(__file__), 'trained_lstm_models.joblib')
            if os.path.exists(lstm_models_path):
                lstm_models = load(lstm_models_path)
                app.logger.info(f"LSTM models loaded successfully. Available profile IDs: {list(lstm_models.keys())}")
            else:
                app.logger.warning(f"LSTM model file not found at {lstm_models_path}. '/predict_lstm' endpoint will not work.")
                lstm_models = {} # Initialize as empty dict if file not found
        except Exception as e:
            app.logger.error(f"Error loading LSTM models: {str(e)}")
            lstm_models = {} # Initialize as empty dict on error

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Check if we have valid JSON in the request
        data = request.get_json()
        if not data:
            return jsonify({'error': 'Invalid JSON or missing request body'}), 400
        
        # Extract profile_id from the request
        profile_id = data.get('profile_id')
        if not profile_id:
            return jsonify({'error': 'Missing profile_id in request'}), 400
            
        # Check if the profile_id exists in our models
        if profile_id not in models:
            return jsonify({'error': f'No model found for profile_id: {profile_id}. Available profiles: {list(models.keys())}'}), 404
        
        # Select the appropriate model data for this profile
        model_data = models[profile_id]
        model = model_data['model']
        scaler = model_data['scaler']
        
        # Get timestamp (use current time if not provided)
        timestamp_str = data.get('timestamp')
        if timestamp_str:
            try:
                # Handle various timestamp formats
                if timestamp_str.endswith('Z'):
                    timestamp_str = timestamp_str[:-1] + '+00:00'
                timestamp = datetime.fromisoformat(timestamp_str)
            except ValueError:
                return jsonify({'error': 'Invalid timestamp format. Use ISO format like 2025-08-15T14:30:00Z'}), 400
        else:
            timestamp = datetime.now()
        
        # Extract basic features (excluding profile_id and timestamp)
        feature_data = {k: v for k, v in data.items() if k not in ['profile_id', 'timestamp']}
        
        # Fetch recent data for temporal features
        recent_data = get_recent_readings_for_temporal_features(profile_id, window_size=5)
        
        # Create temporal features matching the training format
        feature_vector = create_temporal_features_for_prediction(feature_data, recent_data, timestamp)
        
        # Scale the features
        scaled_features = scaler.transform([feature_vector])
        
        # Make prediction
        prediction = model.predict(scaled_features)[0]
        anomaly_score = model.score_samples(scaled_features)[0]
        
        # For Isolation Forest, 1 is normal, -1 is anomaly
        result = "normal" if prediction == 1 else "anomaly"
        
        # Prepare the response
        response = {
            'profile_id': profile_id,
            'timestamp': timestamp.isoformat(),
            'prediction': int(prediction),
            'result': result,
            'score': float(anomaly_score)
        }
        
        # Add explanation if result is an anomaly
        if result == "anomaly":
            response['explanation'] = {
                'message': "This data point was flagged as an anomaly due to unusual patterns in the monitored values.",
                'anomaly_score': float(anomaly_score),
                'feature_values': feature_data,
                'interpretation': "Lower anomaly scores (more negative) indicate stronger anomalies. Values far from typical ranges for this profile contribute to the anomaly detection.",
                'temporal_context': f"Analysis includes temporal patterns from {len(recent_data)} recent readings and time-based features (hour: {timestamp.hour}, day of week: {timestamp.weekday()})."
            }
        
        return jsonify(response), 200
    
    except Exception as e:
        app.logger.error(f"Error during prediction: {str(e)}")
        return jsonify({'error': str(e)}), 500

# LSTM Prediction endpoint
@app.route('/predict_lstm', methods=['POST'])
def predict_lstm():
    try:
        # Check if we have valid JSON in the request
        data = request.get_json()
        if not data:
            return jsonify({'error': 'Invalid JSON or missing request body'}), 400

        # Extract profile_id from the request
        profile_id = data.get('profile_id')
        if not profile_id:
            return jsonify({'error': 'Missing profile_id in request'}), 400

        # Check if the profile_id exists in our LSTM models
        if not lstm_models or profile_id not in lstm_models:
             available_profiles = list(lstm_models.keys()) if lstm_models else []
             return jsonify({'error': f'No LSTM model found for profile_id: {profile_id}. Available profiles: {available_profiles}'}), 404

        # Select the appropriate LSTM model, scaler, AND sensitivity for this profile
        model_data = lstm_models[profile_id]
        lstm_model = model_data.get('model')
        scaler = model_data.get('scaler')
        sensitivity = model_data.get('sensitivity', 'medium') # Default to medium if not found

        if not lstm_model or not scaler:
            app.logger.error(f"Model or scaler missing for profile_id {profile_id} in loaded data.")
            return jsonify({'error': 'Internal server error: Incomplete model data loaded.'}), 500

        # Get timestamp (use current time if not provided)
        timestamp_str = data.get('timestamp')
        if timestamp_str:
            try:
                if timestamp_str.endswith('Z'):
                    timestamp_str = timestamp_str[:-1] + '+00:00'
                timestamp = datetime.fromisoformat(timestamp_str)
            except ValueError:
                return jsonify({'error': 'Invalid timestamp format. Use ISO format like 2025-08-15T14:30:00Z'}), 400
        else:
            timestamp = datetime.now()

        # Fetch recent sequence for LSTM prediction (typically 10 timesteps)
        sequence_data = get_recent_sequence_for_lstm(profile_id, sequence_length=10)
        
        if len(sequence_data) < 10:
            return jsonify({
                'error': f'Insufficient historical data for LSTM prediction. Need at least 10 recent readings, found {len(sequence_data)}. Please ensure the profile has enough historical data.'
            }), 400

        # Extract current input features
        feature_data = {k: v for k, v in data.items() if k not in ['profile_id', 'timestamp']}
        feature_names = ['voltage', 'current', 'power', 'frequency', 'power_factor']
        
        # Prepare sequence data (use historical + current)
        sequence_features = sequence_data[feature_names].values
        current_features = [feature_data.get(name, 0) for name in feature_names]
        
        # Create input sequence: use last 9 historical + current reading
        input_sequence = np.vstack([sequence_features[-9:], current_features])
        
        # Scale the sequence
        scaled_sequence = scaler.transform(input_sequence)
        
        # Reshape for LSTM: [batch_size, timesteps, features]
        lstm_input = scaled_sequence.reshape((1, 10, len(feature_names)))

        # Make prediction
        predicted_scaled_values = lstm_model.predict(lstm_input)[0]
        
        # Inverse transform to original scale
        predicted_values = scaler.inverse_transform(predicted_scaled_values.reshape(1, -1))[0]

        # Define sensitivity thresholds
        thresholds = {
            'low': {'upper': 1.25, 'lower': 0.75},    # +/- 25%
            'medium': {'upper': 1.15, 'lower': 0.85}, # +/- 15%
            'high': {'upper': 1.10, 'lower': 0.90}    # +/- 10%
        }
        current_threshold = thresholds.get(sensitivity, thresholds['medium'])

        # Compare actual vs predicted values
        comparison_results = {}
        overall_status = "within_prediction"
        deviation_summary = {}

        for i, name in enumerate(feature_names):
            actual = current_features[i]
            predicted = predicted_values[i]
            status = "within_prediction"

            # Calculate deviation ratio
            if abs(predicted) > 1e-6:
                deviation = actual / predicted
                if deviation > current_threshold['upper']:
                    status = "significantly_above"
                elif actual > predicted:
                    status = "above_prediction"
                elif deviation < current_threshold['lower']:
                    status = "significantly_below"
                elif actual < predicted:
                    status = "below_prediction"
            elif actual > predicted:
                status = "significantly_above" if actual > 0.1 else "above_prediction"
            elif actual < predicted:
                status = "significantly_below" if actual < -0.1 else "below_prediction"

            comparison_results[name] = {
                'actual': float(actual),
                'predicted': float(predicted),
                'status': status
            }

            # Track significant deviations
            if status in ["significantly_above", "significantly_below"]:
                deviation_summary[name] = {
                    'actual': float(actual), 
                    'predicted': float(predicted), 
                    'status': status
                }
                
                if overall_status == "within_prediction":
                    overall_status = status
                elif overall_status != status and overall_status in ["significantly_above", "significantly_below"]:
                    overall_status = "multiple_significant_deviations"
            elif status in ["above_prediction", "below_prediction"] and overall_status == "within_prediction":
                overall_status = status

        # Prepare response
        response = {
            'profile_id': profile_id,
            'timestamp': timestamp.isoformat(),
            'sensitivity_level': sensitivity,
            'overall_status': overall_status,
            'details': comparison_results,
            'sequence_info': {
                'historical_readings_used': len(sequence_data),
                'sequence_length': 10,
                'latest_historical_timestamp': sequence_data['created_at'].iloc[-1].isoformat() if not sequence_data.empty else None
            },
            'interpretation': {
                'message': f"LSTM prediction for profile {profile_id} using sequence of 10 timesteps with '{sensitivity}' sensitivity.",
                'status_meaning': {
                    'within_prediction': f"Values are within predicted range (+/- {(current_threshold['upper']-1)*100:.0f}%).",
                    'above_prediction': f"Values are slightly above predicted but within '{sensitivity}' threshold.",
                    'below_prediction': f"Values are slightly below predicted but within '{sensitivity}' threshold.",
                    'significantly_above': f"Values are significantly above (+{(current_threshold['upper']-1)*100:.0f}%) predicted values.",
                    'significantly_below': f"Values are significantly below (-{(1-current_threshold['lower'])*100:.0f}%) predicted values.",
                    'multiple_significant_deviations': f"Multiple features showed significant deviations."
                },
                'significant_deviations_summary': deviation_summary if deviation_summary else "No significant deviations detected."
            }
        }

        return jsonify(response), 200

    except Exception as e:
        app.logger.exception(f"Error during LSTM prediction for profile {profile_id if 'profile_id' in locals() else 'unknown'}:")
        return jsonify({'error': f'An unexpected error occurred: {str(e)}'}), 500

# Health check endpoint
@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy'}), 200

