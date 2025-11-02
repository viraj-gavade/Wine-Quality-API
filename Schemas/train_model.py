from typing import Optional, Literal
from pydantic import BaseModel, Field

class TrainingParameters(BaseModel):
    model_type: Literal["classification", "regression"] = Field(
        description="Type of model you want to train"
    )
    epochs: int = Field(
        default=50,
        gt=10,
        lt=150,
        description="Number of epochs to train the model"
    )
    learning_rate: float = Field(
        default=0.001,
        gt=0,
        lt=1,
        description="Learning rate for optimizer"
    )
    optimizer: Literal["Adam", "SGD", "RMSprop"] = Field(
        default="Adam",
        description="Optimizer to be used for training"
    )
    loss_function: Literal["CrossEntropyLoss", "MSELoss", "MAELoss"] = Field(
        default="CrossEntropyLoss",
        description="Loss function used for training"
    )
    output_feature: Literal["type", "quality"] = Field(default="quality", description="Target column for training (type or quality)")
    
    model_file_name: str = Field(default="wine_model.pth", description="Name of the model file to be saved (with .pth extension)")



