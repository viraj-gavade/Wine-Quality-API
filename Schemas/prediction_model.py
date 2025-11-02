from pydantic import BaseModel, Field
from typing import Literal

class PredictionParameters(BaseModel):
    model_type: Literal["classification", "regression"] = Field(
        ...,
        description="Type of model to use: classification or regression"
    )
    model_file_name: str = Field(
        ...,
        description="Filename of the trained model to be loaded"
    )
    input_data: list[list[float]] = Field(
        ...,
        description="List of feature vectors for prediction (2D array-like)"
    )
    device: Literal["cpu", "cuda"] = Field(
        default="cpu",
        description="Device to run prediction on (CPU or CUDA)"
    )
