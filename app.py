from flask import Flask, request, jsonify
from joblib import load
import os
import numpy as np

app = Flask(__name__)
models = None  # Global variable to hold the loaded Isolation Forest models dictionary
lstm_models = None # Global variable to hold the loaded LSTM models dictionary

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
        
        # Select the appropriate model for this profile
        model = models[profile_id]
        
        # Remove profile_id from features and keep only the model input features
        feature_data = {k: v for k, v in data.items() if k != 'profile_id'}
        
        # Convert input data to the format expected by the model
        features = [list(feature_data.values())]  # Convert dictionary values to a list of lists
        
        # Make prediction using the selected model
        prediction = model.predict(features)[0]  # Get the first element of the prediction
        
        # Get anomaly score (negative means more anomalous)
        anomaly_score = model.score_samples(features)[0]
        
        # For Isolation Forest, 1 is normal, -1 is anomaly
        result = "normal" if prediction == 1 else "anomaly"
        
        # Prepare the response
        response = {
            'profile_id': profile_id,
            'prediction': int(prediction),
            'result': result,
            'score': float(anomaly_score)
        }
        
        # Add explanation if result is an anomaly
        if result == "anomaly":
            # Extract the feature names and values
            feature_names = list(feature_data.keys())
            feature_values = list(feature_data.values())
            
            # Calculate contribution of each feature to the anomaly
            # This is a simplified approach - for each feature, check how far it is from typical values
            # Ideally, we would have stored the feature means during training
            feature_contributions = {}
            
            for i, feature_name in enumerate(feature_names):
                feature_contributions[feature_name] = feature_values[i]
            
            # Add explanation to the response
            response['explanation'] = {
                'message': "This data point was flagged as an anomaly due to unusual patterns in the monitored values.",
                'anomaly_score': float(anomaly_score),
                'feature_values': feature_contributions,
                'interpretation': "Lower anomaly scores (more negative) indicate stronger anomalies. Values far from typical ranges for this profile contribute to the anomaly detection."
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

        # Select the appropriate LSTM model for this profile
        lstm_model = lstm_models[profile_id]

        # Remove profile_id from features and keep only the model input features
        feature_data = {k: v for k, v in data.items() if k != 'profile_id'}
        feature_names = list(feature_data.keys())
        feature_values = list(feature_data.values())

        # Convert input data to the format expected by the LSTM model
        # IMPORTANT: LSTM models usually require sequential data (e.g., shape [batch_size, timesteps, features])
        # Assuming the loaded LSTM model's predict method handles a single timestep input appropriately
        # or expects a specific shape. This might need adjustment based on how the LSTM was trained.
        # For this example, we'll assume it takes a shape like [1, 1, num_features] or simply [1, num_features]
        # Let's try reshaping to [1, num_features] which is common for simpler prediction tasks.
        # If the model expects sequences, this needs significant change.
        try:
            features_array = np.array([feature_values], dtype=float) # Shape [1, num_features]
            # If LSTM expects sequence like [1, 1, num_features]:
            # features_array = np.array([feature_values], dtype=float).reshape(1, 1, -1)
        except ValueError:
             return jsonify({'error': 'Input features must be numeric'}), 400


        # Make prediction using the selected LSTM model
        # Assuming the model predicts the 'expected' or 'threshold' values for the next step
        predicted_values = lstm_model.predict(features_array)[0] # Get the first (and only) prediction

        # Ensure predicted_values is iterable and has same length as feature_names
        if not hasattr(predicted_values, '__iter__') or len(predicted_values) != len(feature_names):
             app.logger.error(f"LSTM model for profile {profile_id} produced unexpected output shape.")
             return jsonify({'error': 'Internal server error: LSTM model prediction format mismatch.'}), 500


        # Compare actual values with predicted values/thresholds
        comparison_results = {}
        overall_status = "within_prediction" # Default status
        high_features = {}

        for i, name in enumerate(feature_names):
            actual = feature_values[i]
            predicted = predicted_values[i]
            is_above = actual > predicted
            comparison_results[name] = {
                'actual': float(actual),
                'predicted': float(predicted),
                'status': 'above_prediction' if is_above else 'within_prediction'
            }
            if is_above:
                overall_status = "above_prediction" # Change status if any feature is above
                high_features[name] = {'actual': float(actual), 'predicted': float(predicted)}

        # Prepare the response
        response = {
            'profile_id': profile_id,
            'overall_status': overall_status,
            'details': comparison_results,
            'interpretation': {
                'message': f"Comparison of input data against LSTM predictions for profile {profile_id}.",
                'status_meaning': {
                    'within_prediction': "Input values are within the predicted range or below the predicted threshold.",
                    'above_prediction': "One or more input values are above the predicted range or threshold, indicating a potential deviation."
                },
                'high_features_summary': high_features if high_features else "No features were significantly above predicted values."
            }
        }

        return jsonify(response), 200

    except Exception as e:
        app.logger.error(f"Error during LSTM prediction: {str(e)}")
        # Consider more specific error handling based on potential exceptions
        return jsonify({'error': f'An unexpected error occurred: {str(e)}'}), 500

# Health check endpoint
@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy'}), 200

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)