
import torch.nn as nn
import torch 
from torch.utils.data import Dataset,DataLoader

class CustomDataset(Dataset):
    def __init__(self,features , labels ):
        self.features = torch.tensor(features,dtype=torch.float32)
        self.labels = torch.tensor(labels,dtype=torch.long)

    def __len__(self):
        return len(self.features)
    

    def __getitem__(self, index):
        return self.features[index] , self.labels[index]
        