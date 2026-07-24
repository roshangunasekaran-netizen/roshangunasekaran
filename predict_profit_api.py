import pandas as pd
import numpy as np
import joblib
from flask import Flask, request, jsonify
import os

app = Flask(__name__)

# --- Load Models and Global Variables ---
# IMPORTANT: These paths assume the models are in the same directory as the app.py file
# For Render deployment, you'll need to ensure these files are accessible.

w_model_path = 'wheat_yield_linear_model.joblib'
m_model_path = 'maize_yield_linear_model.joblib'
s_model_path = 'soybeans_yield_linear_model.joblib'
r_model_path = 'rice_yield_linear_model.joblib'
sg_model_path = 'sorghum_yield_linear_model.joblib'
p_model_path = 'potatoes_yield_linear_model.joblib'
cv_model_path = 'cassava_yield_linear_model.joblib'
sp_model_path = 'sweet_potatoes_yield_linear_model.joblib'

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

# Load Soybeans Yield Model
try:
    soybeans_yield_model = joblib.load(s_model_path)
    print(f"Loaded Soybeans yield model from {s_model_path}")
except FileNotFoundError:
    print(f"Error: Soybeans model not found at {s_model_path}. Please ensure it's present.")
    soybeans_yield_model = None

# Load Rice Yield Model
try:
    rice_yield_model = joblib.load(r_model_path)
    print(f"Loaded Rice yield model from {r_model_path}")
except FileNotFoundError:
    print(f"Error: Rice model not found at {r_model_path}. Please ensure it's present.")
    rice_yield_model = None

# Load Sorghum Yield Model
try:
    sorghum_yield_model = joblib.load(sg_model_path)
    print(f"Loaded Sorghum yield model from {sg_model_path}")
except FileNotFoundError:
    print(f"Error: Sorghum model not found at {sg_model_path}. Please ensure it's present.")
    sorghum_yield_model = None

# Load Potatoes Yield Model
try:
    potatoes_yield_model = joblib.load(p_model_path)
    print(f"Loaded Potatoes yield model from {p_model_path}")
except FileNotFoundError:
    print(f"Error: Potatoes model not found at {p_model_path}. Please ensure it's present.")
    potatoes_yield_model = None

# Load Cassava Yield Model
try:
    cassava_yield_model = joblib.load(cv_model_path)
    print(f"Loaded Cassava yield model from {cv_model_path}")
except FileNotFoundError:
    print(f"Error: Cassava model not found at {cv_model_path}. Please ensure it's present.")
    cassava_yield_model = None

# Load Sweet potatoes Yield Model
try:
    sweet_potatoes_yield_model = joblib.load(sp_model_path)
    print(f"Loaded Sweet potatoes yield model from {sp_model_path}")
except FileNotFoundError:
    print(f"Error: Sweet potatoes model not found at {sp_model_path}. Please ensure it's present.")
    sweet_potatoes_yield_model = None

# Placeholder for average prices and costs
# From previous analysis:
# Overall Average Price (INR/kg): 19.13 (Wheat)
# Overall Average Maize Price (INR/kg): 17.68 (Maize)
# Overall Average Soybeans Price (INR/kg): 34.23 (Soybeans)
# Overall Average Rice Price (INR/kg): 28.28 (Rice)
# Overall Average Sorghum Price (INR/kg): 18.17 (Sorghum)
# Overall Average Potatoes Price (INR/kg): 7.62 (Potatoes)
# Overall Average Cassava Price (INR/kg): 23.67 (Cassava)
# Overall Average Sweet potatoes Price (INR/kg): 19.62 (Sweet potatoes)

overall_avg_price_wheat = 19.134
overall_avg_price_maize = 17.676363636363636
overall_avg_price_soybeans = 34.225
overall_avg_price_rice = 28.276415094339626
overall_avg_price_sorghum = 18.166666666666668
overall_avg_price_potatoes = 7.623097345132743
overall_avg_price_cassava = 23.67142857142857
overall_avg_sweet_potatoes_price = 19.61875
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
            'avg_price': overall_avg_maize_price
        },
        'soybeans': {
            'model': soybeans_yield_model,
            'avg_price': overall_avg_price_soybeans
        },
        'rice': {
            'model': rice_yield_model,
            'avg_price': overall_avg_rice_price
        },
        'sorghum': {
            'model': sorghum_yield_model,
            'avg_price': overall_avg_price_sorghum
        },
        'potatoes': {
            'model': potatoes_yield_model,
            'avg_price': overall_avg_potatoes_price
        },
        'cassava': {
            'model': cassava_yield_model,
            'avg_price': overall_avg_cassava_price
        },
        'sweet potatoes': {
            'model': sweet_potatoes_yield_model,
            'avg_price': overall_avg_sweet_potatoes_price
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
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
