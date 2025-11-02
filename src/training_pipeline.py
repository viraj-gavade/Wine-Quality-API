import os
import torch
from torch.utils.data import DataLoader

def train_model(
    epochs: int,
    train_loader: DataLoader,
    optimizer,
    loss_function,
    model,
    model_name="model.pth",
    task_type="classification",  
):
    save_dir = "Models/trained"
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, model_name)

    print(f"Training started for {epochs} epochs...")

    for epoch in range(epochs):
        model.train()
        total_epoch_loss = 0

        for batch_features, batch_labels in train_loader:
            batch_features = batch_features.float()
            if task_type == "classification":
                batch_labels = batch_labels.long()
            else:
                batch_labels = batch_labels.float()

            y_pred = model(batch_features)
            loss = loss_function(y_pred, batch_labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_epoch_loss += loss.item()

        avg_loss = total_epoch_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{epochs}] | Average Loss: {avg_loss:.4f}")

    torch.save(model.state_dict(), save_path)
    print(f"Model saved at: {save_path}")
