from fastapi import APIRouter
from controllers.model_controllers import train_neural_network ,test_neural_network ,predict_output
from Schemas.train_model import TrainingParameters
from Schemas.prediction_model import PredictionParameters
ModelRouter = APIRouter()



@ModelRouter.post('/train/')
def train_model(params : TrainingParameters):
    return train_neural_network(params)

@ModelRouter.post('/test/')
def evaluate_model(params : TrainingParameters):
    return test_neural_network(params)

@ModelRouter.post('/predict')
def predict(params :PredictionParameters):
    return predict_output(params)





