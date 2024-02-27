from flask import Flask, request, jsonify
import mlflow.sklearn
import numpy as np

app = Flask(__name__)

# Load the model from MLflow
model_version = "2"  # Change this to the desired version
model = mlflow.sklearn.load_model(f"models:/mobile_classification_model/{model_version}")

# Number of expected features
expected_num_features = 21

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get features from JSON request
        features = request.json.get('features', [])

        # Check the number of features
        if len(features) != expected_num_features:
            raise ValueError(f"Expected {expected_num_features} features, but received {len(features)} features.")

        # Make predictions
        prediction = model.predict([features])[0]

        # Return the prediction as JSON
        return jsonify({'prediction': int(prediction)})
    
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)
