import joblib
import pandas as pd
from flask import Flask, request, jsonify
import os

# Define file paths for loading
MODEL_FILE = 'crop_recommendation_model.joblib'
LABEL_ENCODER_FILE = 'crop_label_encoder.joblib'

# Ensure these files are present in the same directory as app.py when deployed
# In Colab, they are in /content/, so we refer to them directly.
# In Render, they would need to be part of your uploaded project files.

# Load the trained model and label encoder
try:
    loaded_model = joblib.load(MODEL_FILE)
    loaded_le = joblib.load(LABEL_ENCODER_FILE)
    print("Model and Label Encoder loaded successfully!")
except Exception as e:
    print(f"Error loading model or label encoder: {e}")
    # In a production app, you might want to exit or log a critical error

app = Flask(__name__)

def predict_crop(N, P, K, temperature, humidity, ph, rainfall):
    # Create a DataFrame from the input
    input_data = pd.DataFrame([{
        'N': N,
        'P': P,
        'K': K,
        'temperature': temperature,
        'humidity': humidity,
        'ph': ph,
        'rainfall': rainfall
    }])

    # Make a prediction using the loaded model
    predicted_label_encoded = loaded_model.predict(input_data)
    prediction_probabilities = loaded_model.predict_proba(input_data)

    # Get the confidence for the predicted crop
    predicted_crop_index = predicted_label_encoded[0]
    confidence = prediction_probabilities[0, predicted_crop_index] * 100

    # Decode the predicted label
    predicted_crop = loaded_le.inverse_transform(predicted_label_encoded)

    return predicted_crop[0], confidence

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    
    N = data.get('N')
    P = data.get('P')
    K = data.get('K')
    temperature = data.get('temperature')
    humidity = data.get('humidity')
    ph = data.get('ph')
    rainfall = data.get('rainfall')
    
    if None in [N, P, K, temperature, humidity, ph, rainfall]:
        return jsonify({'error': 'Missing one or more required parameters. Ensure N, P, K, temperature, humidity, ph, rainfall are provided.'}), 400
        
    try:
        predicted_crop_name, confidence = predict_crop(N, P, K, temperature, humidity, ph, rainfall)
        return jsonify({
            'predicted_crop': predicted_crop_name,
            'confidence': f"{confidence:.2f}%"
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # For local testing, you can use app.run()
    # On Render, the web server (like Gunicorn) will handle running the app.
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
