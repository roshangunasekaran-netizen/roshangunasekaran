from contextlib import asynccontextmanager
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

# Dictionary to store your model
ml_models = {}

# Define the input data structure for the prediction endpoint
class ProfitPredictionInput(BaseModel):
    Year: int
    Area_harvested_ha: float
    Production_tonnes: float
    Yield_hg_ha: float
    Average_Price_USD: float
    Crop: str

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load your model on startup
    ml_models["my_model"] = joblib.load("/content/profit_prediction_model.joblib")
    yield
    # Clean up on shutdown
    ml_models.clear()

app = FastAPI(lifespan=lifespan)

@app.post("/predict")
async def predict(input_data: ProfitPredictionInput):
    input_df = pd.DataFrame([input_data.dict()])
    predicted_profit = ml_models["my_model"].predict(input_df)
    return {"predicted_profit_usd": predicted_profit[0]}
