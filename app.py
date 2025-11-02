from fastapi import FastAPI
import torch

# ---- Imports ----
from src.classsification_model import WineTypePredictionNN
from src.training_pipeline import train_model
from utils.parameters import Parameters
from utils.data_preprocessing import process_data
from utils.data_loader import load_data
from utils.load_model import load_model
from routes.model_routes import ModelRouter


params = Parameters()


model = WineTypePredictionNN(num_features=params.num_features)


optimizer = torch.optim.Adam(model.parameters(), lr=params.learning_rate)
loss_function = params.loss_function

app = FastAPI()


app.include_router(ModelRouter,prefix='/model',tags=['model'])
@app.get("/")
def home():
    return {"message": "✅ FastAPI is running successfully!"}





