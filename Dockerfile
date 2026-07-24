FROM python:3.9-slim

WORKDIR /app

# Copy requirements file and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY predict_profit_api.py .

# Copy the trained models
COPY wheat_yield_linear_model.joblib .
COPY maize_yield_linear_model.joblib .
COPY soybeans_yield_linear_model.joblib .
COPY rice_yield_linear_model.joblib .
COPY sorghum_yield_linear_model.joblib .
COPY potatoes_yield_linear_model.joblib .
COPY cassava_yield_linear_model.joblib .
COPY sweet_potatoes_yield_linear_model.joblib .

EXPOSE 5000

CMD ["gunicorn", "--bind", "0.0.0.0:5000", "predict_profit_api:app"]

EXPOSE 10000

CMD ["sh", "-c", "gunicorn --bind 0.0.0.0:${PORT} predict_profit_api:app"]
