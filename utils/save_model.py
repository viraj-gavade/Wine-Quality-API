import torch

def save_model(model, path):
    """Save a PyTorch model's state dict."""
    torch.save(model.state_dict(), path)
    print(f"✅ Model saved at: {path}")