import pandas as pd
import numpy as np
import joblib
from flask import Flask, request, jsonify
import os

app = Flask(__name__)

# --- Load Models and Global Variables ---
# IMPORTANT: These paths assume the models are in the same directory as the app.py file
# For Render deployment, you'll need to ensure these files are accessible.
# You might need to adjust the paths or host the models in a cloud storage service
# and download them during deployment.

w_model_path = 'wheat_yield_linear_model.joblib'
m_model_path = 'maize_yield_linear_model.joblib'

# Load Wheat Yield Model
try:
    yield_model = joblib.load(w_model_path)
    print(f"Loaded Wheat yield model from {w_model_path}")
except FileNotFoundError:
    print(f"Error: Wheat model not found at {w_model_path}. Please ensure it's present.")
    yield_model = None

# Load Maize Yield Model
try:
    maize_yield_model = joblib.load(m_model_path)
    print(f"Loaded Maize yield model from {m_model_path}")
except FileNotFoundError:
    print(f"Error: Maize model not found at {m_model_path}. Please ensure it's present.")
    maize_yield_model = None

# Placeholder for average prices and costs
# In a real deployment, these would likely come from a database or a configuration file.
# For now, we'll hardcode the values derived from our analysis.

# From previous analysis:
# Overall Average Yield (kg/acre): 1074.31 (Wheat)
# Overall Average Price (INR/kg): 19.13 (Wheat)
# Overall Average Farming Cost (INR/acre): 197507.69 (General Proxy)
# Overall Average Maize Price (INR/kg): 17.68 (Maize)

overall_avg_price_wheat = 19.134
overall_avg_price_maize = 17.676363636363636
overall_avg_farming_cost = 197507.69148148145

# --- Prediction Function (Adapted from Notebook) ---
def predict_profit_for_crop(crop_name, year, average_rain_fall_mm_per_year, pesticides_tonnes, avg_temp, land_area_acres):
    crop_name_lower = crop_name.lower()

    crop_configs = {
        'wheat': {
            'model': yield_model,
            'avg_price': overall_avg_price_wheat
        },
        'maize': {
            'model': maize_yield_model,
            'avg_price': overall_avg_price_maize
        }
    }

    if crop_name_lower not in crop_configs:
        return {"error": f"Prediction for '{crop_name}' is not currently supported. Supported crops: {', '.join(crop_configs.keys())}"}

    config = crop_configs[crop_name_lower]
    current_yield_model = config['model']
    current_avg_price = config['avg_price']

    if current_yield_model is None:
        return {"error": f"Model for {crop_name} is not loaded. Cannot make prediction."}

    input_features = np.array([[year, average_rain_fall_mm_per_year, pesticides_tonnes, avg_temp]])

    # Predict yield
    predicted_yield_kg_per_acre = current_yield_model.predict(input_features)[0]

    # Calculate total revenue
    total_revenue_inr = predicted_yield_kg_per_acre * current_avg_price * land_area_acres

    # Calculate total cost
    total_cost_inr = overall_avg_farming_cost * land_area_acres

    # Calculate predicted profit
    predicted_profit_inr = total_revenue_inr - total_cost_inr

    return {
        "crop_name": crop_name,
        "land_area_acres": land_area_acres,
        "predicted_yield_kg_per_acre": round(float(predicted_yield_kg_per_acre), 2),
        "estimated_revenue_inr": round(float(total_revenue_inr), 2),
        "estimated_cost_inr_proxy": round(float(total_cost_inr), 2),
        "predicted_profit_inr": round(float(predicted_profit_inr), 2)
    }

# --- API Endpoint ---
@app.route('/predict_profit', methods=['POST'])
def predict_profit():
    data = request.get_json(force=True)

    required_fields = ['crop_name', 'year', 'average_rain_fall_mm_per_year', 'pesticides_tonnes', 'avg_temp', 'land_area_acres']
    if not all(field in data for field in required_fields):
        return jsonify({"error": "Missing one or more required fields"}), 400

    crop_name = data['crop_name']
    year = data['year']
    average_rain_fall_mm_per_year = data['average_rain_fall_mm_per_year']
    pesticides_tonnes = data['pesticides_tonnes']
    avg_temp = data['avg_temp']
    land_area_acres = data['land_area_acres']

    result = predict_profit_for_crop(
        crop_name=crop_name,
        year=year,
        average_rain_fall_mm_per_year=average_rain_fall_mm_per_year,
        pesticides_tonnes=pesticides_tonnes,
        avg_temp=avg_temp,
        land_area_acres=land_area_acres
    )

    if "error" in result:
        return jsonify(result), 400 # Return 400 for bad request if there's an error
    return jsonify(result)

# --- Health Check Endpoint ---
@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy'}), 200

if __name__ == '__main__':
    # For local testing, you can run:
    # flask run --host=0.0.0.0 --port=5000
    # For Render, it will automatically use the PORT environment variable
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
