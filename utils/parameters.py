import torch
import torch.nn as nn

class Parameters:
    model : str
    learning_rate : float = 0.001
    loss_function = nn.CrossEntropyLoss()
    epochs = 100
    num_features = 12
    
