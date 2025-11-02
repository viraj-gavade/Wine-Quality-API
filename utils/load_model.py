import torch
def load_model(model, path):
    model.load_state_dict(torch.load(path, map_location='cpu'))
    model.eval()
    return model