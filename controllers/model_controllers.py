import os
import torch
import logging
from datetime import datetime
from src.training_pipeline import train_model
from utils.data_loader import load_data
from utils.data_preprocessing import process_data
from Schemas.train_model import TrainingParameters
from src.classsification_model import WineTypePredictionNN
from src.regression_model import WineQualityRegressionNN
from src.testing_pipeline import evaluate_model
from Schemas.prediction_model import PredictionParameters

# Use environment variables for paths so containers / platforms can override them
DATA_PATH = os.environ.get("DATA_PATH", "data/proccessed_winequality.csv")
MODEL_DIR = os.environ.get("MODEL_DIR", "Models/trained")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("logs/training.log"),
        logging.StreamHandler()
    ]
)

def train_neural_network(params: TrainingParameters):
    try:
        logging.info("Starting training process...")
        logging.info(f"Received parameters: {params}")

        if params.model_type == "classification":
            target_col = "type"
        elif params.model_type == "regression":
            target_col = "quality"
        else:
            raise ValueError("Invalid model_type. Choose 'classification' or 'regression'.")

        # Load and prepare data
        df = load_data(DATA_PATH)
        train_loader, _ = process_data(df, output_feature=target_col)

        # Create model
        if params.model_type == "classification":
            model = WineTypePredictionNN(num_features=12)
        else:
            model = WineQualityRegressionNN(num_features=12)

        # Optimizer
        if params.optimizer == "Adam":
            optimizer = torch.optim.Adam(model.parameters(), lr=params.learning_rate)
        elif params.optimizer == "SGD":
            optimizer = torch.optim.SGD(model.parameters(), lr=params.learning_rate)
        elif params.optimizer == "RMSprop":
            optimizer = torch.optim.RMSprop(model.parameters(), lr=params.learning_rate)
        else:
            raise ValueError("Unsupported optimizer selected.")

        # Loss validation
        if params.model_type == "classification" and params.loss_function != "CrossEntropyLoss":
            raise ValueError("Classification requires CrossEntropyLoss.")
        if params.model_type == "regression" and params.loss_function not in ["MSELoss", "MAELoss"]:
            raise ValueError("Regression requires MSELoss or MAELoss.")

        if params.loss_function == "CrossEntropyLoss":
            loss_function = torch.nn.CrossEntropyLoss()
        elif params.loss_function == "MSELoss":
            loss_function = torch.nn.MSELoss()
        elif params.loss_function == "MAELoss":
            loss_function = torch.nn.L1Loss()
        else:
            raise ValueError("Invalid loss function specified.")

        # Name for saved model
        model_save_name = f"{params.model_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pth"
        model_path = f"{MODEL_DIR}/{model_save_name}"

        # Run training pipeline (this function is defined in src.training_pipeline)
        train_model(
            epochs=params.epochs,
            train_loader=train_loader,
            optimizer=optimizer,
            loss_function=loss_function,
            model=model,
            model_name=params.model_file_name,
            task_type=params.model_type
        )

        return {
            "status": "Training completed successfully",
            "model_type": params.model_type,
            "optimizer": params.optimizer,
            "loss_function": params.loss_function,
            "epochs": params.epochs,
            "learning_rate": params.learning_rate,
            "model_path": model_path
        }

    except Exception as e:
        logging.error(f"Training failed: {e}", exc_info=True)
        return {"status": "Training failed", "error": str(e)}


def test_neural_network(params: TrainingParameters):
    try:
        logging.info("Starting model evaluation...")
        logging.info(f"Received parameters: {params}")

        if params.model_type == "classification":
            target_col = "type"
        elif params.model_type == "regression":
            target_col = "quality"
        else:
            raise ValueError("Invalid model_type. Choose 'classification' or 'regression'.")

        df = load_data(DATA_PATH)
        _, test_loader = process_data(df, output_feature=target_col)

        if params.model_type == "classification":
            model = WineTypePredictionNN(num_features=12)
        else:
            model = WineQualityRegressionNN(num_features=12)

        model_path = f"{MODEL_DIR}/{params.model_file_name}"
        model.load_state_dict(torch.load(model_path, map_location="cpu"))
        model.eval()

        if params.model_type == "classification" and params.loss_function != "CrossEntropyLoss":
            raise ValueError("Classification requires CrossEntropyLoss.")
        if params.model_type == "regression" and params.loss_function not in ["MSELoss", "MAELoss"]:
            raise ValueError("Regression requires MSELoss or MAELoss.")

        if params.loss_function == "CrossEntropyLoss":
            loss_function = torch.nn.CrossEntropyLoss()
        elif params.loss_function == "MSELoss":
            loss_function = torch.nn.MSELoss()
        elif params.loss_function == "MAELoss":
            loss_function = torch.nn.L1Loss()
        else:
            raise ValueError("Invalid loss function specified.")

        results = evaluate_model(
            model=model,
            dataloader=test_loader,
            task_type=params.model_type,
            loss_function=loss_function
        )

        return {
            "status": "Model evaluation completed successfully",
            "model_type": params.model_type,
            "model_file_name": params.model_file_name,
            "results": results
        }

    except Exception as e:
        logging.error(f"Model evaluation failed: {e}", exc_info=True)
        return {"status": "Model evaluation failed", "error": str(e)}


def predict_output(params: PredictionParameters):
    try:
        logging.info(f"Received prediction request: {params}")

        # load the model
        if params.model_type == "classification":
            model = WineTypePredictionNN(num_features=len(params.input_data[0]))
        elif params.model_type == "regression":
            model = WineQualityRegressionNN(num_features=len(params.input_data[0]))
        else:
            raise ValueError("Invalid model type")

        model_path = f"{MODEL_DIR}/{params.model_file_name}"
        model.load_state_dict(torch.load(model_path, map_location=params.device))
        model.to(params.device)
        model.eval()

        # convert input to tensor
        inputs = torch.tensor(params.input_data, dtype=torch.float32).to(params.device)

        with torch.no_grad():
            outputs = model(inputs)

            if params.model_type == "classification":
                predictions = torch.argmax(outputs, dim=1).cpu().numpy().tolist()
            else:
                predictions = outputs.cpu().numpy().flatten().tolist()

        logging.info(f"Predictions: {predictions}")
        return {
            "status": "Prediction successful",
            "model_type": params.model_type,
            "predictions": predictions
        }

    except Exception as e:
        logging.error(f"Prediction failed: {e}", exc_info=True)
        return {
            "status": "Prediction failed",
            "error": str(e)
        }