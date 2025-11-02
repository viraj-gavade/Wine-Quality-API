import torch.nn as nn
import torch



class WineTypePredictionNN(nn.Module):
    def __init__(self, num_features):
        
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(num_features,32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(32,16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(16,8),
            nn.BatchNorm1d(8),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(8,2)
        )

    def forward(self, x):
        return self.network(x)