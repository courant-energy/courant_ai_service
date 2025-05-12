from flask import Flask, request, jsonify
from joblib import load
import os
import numpy as np
from sklearn.preprocessing import MinMaxScaler

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

        # Select the appropriate LSTM model, scaler, AND sensitivity for this profile
        model_data = lstm_models[profile_id]
        lstm_model = model_data.get('model')
        scaler = model_data.get('scaler')
        sensitivity = model_data.get('sensitivity', 'medium') # Default to medium if not found

        if not lstm_model or not scaler:
            app.logger.error(f"Model or scaler missing for profile_id {profile_id} in loaded data.")
            return jsonify({'error': 'Internal server error: Incomplete model data loaded.'}), 500

        # Define sensitivity thresholds (adjust percentages as needed)
        thresholds = {
            'low': {'upper': 1.25, 'lower': 0.75},    # +/- 25%
            'medium': {'upper': 1.15, 'lower': 0.85}, # +/- 15%
            'high': {'upper': 1.10, 'lower': 0.90}    # +/- 10%
        }
        current_threshold = thresholds.get(sensitivity, thresholds['medium']) # Use medium if sensitivity value is unexpected

        # Remove profile_id from features and keep only the model input features
        feature_data = {k: v for k, v in data.items() if k != 'profile_id'}
        feature_names = list(feature_data.keys())
        feature_values = list(feature_data.values())

        # Convert input data to numpy array
        try:
            features_array = np.array([feature_values], dtype=float) # Shape [1, num_features]
        except ValueError:
             return jsonify({'error': 'Input features must be numeric'}), 400

        # Scale the input features using the loaded scaler
        try:
            scaled_features = scaler.transform(features_array)
        except ValueError as e:
             app.logger.error(f"Error scaling features for profile {profile_id}: {e}")
             return jsonify({'error': f'Feature mismatch or scaling error: {e}'}), 400
        except Exception as e:
             app.logger.error(f"Unexpected error during scaling for profile {profile_id}: {e}")
             return jsonify({'error': 'Internal server error during data scaling.'}), 500


        # Reshape features for LSTM input: [batch_size, timesteps, features]
        # NOTE: The model was trained on sequences (e.g., 10 timesteps).
        # Predicting based on a single timestep might not be optimal.
        # This assumes the model can handle a sequence of length 1.
        # A better approach might involve fetching recent history or redesigning the model/endpoint.
        lstm_input = scaled_features.reshape((1, 1, scaled_features.shape[1]))

        # Make prediction using the selected LSTM model
        predicted_scaled_values = lstm_model.predict(lstm_input)[0] # Shape [num_features]

        # Inverse transform the prediction to the original scale
        try:
            predicted_values = scaler.inverse_transform(predicted_scaled_values.reshape(1, -1))[0]
        except Exception as e:
             app.logger.error(f"Error inverse transforming prediction for profile {profile_id}: {e}")
             return jsonify({'error': 'Internal server error during prediction scaling.'}), 500

        # Ensure predicted_values has same length as feature_names
        if len(predicted_values) != len(feature_names):
             app.logger.error(f"LSTM model for profile {profile_id} produced unexpected output shape after scaling.")
             return jsonify({'error': 'Internal server error: LSTM model prediction format mismatch.'}), 500

        # Compare actual values with predicted values using sensitivity thresholds
        comparison_results = {}
        overall_status = "within_prediction" # Default status
        deviation_summary = {}

        for i, name in enumerate(feature_names):
            actual = feature_values[i]
            predicted = predicted_values[i]
            status = "within_prediction"
            deviation = None

            # Avoid division by zero if predicted is 0 or very close to 0
            if abs(predicted) > 1e-6:
                deviation = actual / predicted
                if deviation > current_threshold['upper']:
                    status = "significantly_above"
                elif actual > predicted: # Above predicted but within threshold
                    status = "above_prediction"
                elif deviation < current_threshold['lower']:
                    status = "significantly_below"
                elif actual < predicted: # Below predicted but within threshold
                    status = "below_prediction"
            elif actual > predicted: # Handle predicted near zero case simply
                 status = "significantly_above" if actual > 0.1 else "above_prediction" # Arbitrary small threshold
            elif actual < predicted:
                 status = "significantly_below" if actual < -0.1 else "below_prediction"

            comparison_results[name] = {
                'actual': float(actual),
                'predicted': float(predicted),
                'status': status
            }

            # Track the most significant deviation for overall status and summary
            if status == "significantly_above" or status == "significantly_below":
                deviation_summary[name] = {'actual': float(actual), 'predicted': float(predicted), 'status': status}
                # Prioritize significant deviations for overall status
                if overall_status != "significantly_above" and overall_status != "significantly_below":
                     overall_status = status
                elif status == "significantly_above" and overall_status == "significantly_below":
                     overall_status = "multiple_significant_deviations" # Indicate both types occurred
                elif status == "significantly_below" and overall_status == "significantly_above":
                     overall_status = "multiple_significant_deviations"
            elif (status == "above_prediction" or status == "below_prediction") and overall_status == "within_prediction":
                 overall_status = status # Update to minor deviation if only within_prediction so far

        # Prepare the response
        response = {
            'profile_id': profile_id,
            'sensitivity_level': sensitivity,
            'overall_status': overall_status,
            'details': comparison_results,
            'interpretation': {
                'message': f"Comparison of input data against LSTM predictions for profile {profile_id} using '{sensitivity}' sensitivity.",
                'status_meaning': {
                    'within_prediction': f"Input values are within the predicted range (+/- { (current_threshold['upper']-1)*100:.0f}%).",
                    'above_prediction': f"Input values are slightly above the predicted value but within the '{sensitivity}' threshold.",
                    'below_prediction': f"Input values are slightly below the predicted value but within the '{sensitivity}' threshold.",
                    'significantly_above': f"One or more input values are significantly above (+{ (current_threshold['upper']-1)*100:.0f}%) the predicted value based on '{sensitivity}' sensitivity.",
                    'significantly_below': f"One or more input values are significantly below (-{ (1-current_threshold['lower'])*100:.0f}%) the predicted value based on '{sensitivity}' sensitivity.",
                    'multiple_significant_deviations': f"Multiple input values showed significant deviations (both above and below) based on '{sensitivity}' sensitivity."
                },
                'significant_deviations_summary': deviation_summary if deviation_summary else "No features showed significant deviations from predicted values."
            }
        }

        return jsonify(response), 200

    except Exception as e:
        # Log the full traceback for better debugging
        app.logger.exception(f"Error during LSTM prediction for profile {profile_id if 'profile_id' in locals() else 'unknown'}:")
        return jsonify({'error': f'An unexpected error occurred: {str(e)}'}), 500

# Health check endpoint
@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy'}), 200

