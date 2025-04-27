from flask import Flask, request, jsonify
from joblib import load
import os

app = Flask(__name__)
model = None  # Global variable to hold the loaded model

# For newer Flask versions (2.0+) which don't support @app.before_first_request
@app.before_request
def load_model_if_needed():
    global model
    if model is None:
        model_path = os.path.join(os.path.dirname(__file__), 'model.joblib')
        model = load(model_path)
        app.logger.info("Model loaded successfully")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Check if we have valid JSON in the request
        data = request.get_json()
        if not data:
            return jsonify({'error': 'Invalid JSON or missing request body'}), 400
        
        # Convert input data to the format expected by the model
        # Assuming data is a dictionary where keys match the feature names
        features = [list(data.values())]  # Convert dictionary values to a list of lists
        
        # Make prediction using the loaded model
        prediction = model.predict(features)[0]  # Get the first element of the prediction
        
        # Return the prediction as JSON
        return jsonify({'prediction': prediction.tolist() if hasattr(prediction, 'tolist') else prediction}), 200
    
    except Exception as e:
        app.logger.error(f"Error during prediction: {str(e)}")
        return jsonify({'error': str(e)}), 500

# Health check endpoint
@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy'}), 200

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)