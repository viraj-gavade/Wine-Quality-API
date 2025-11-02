import torch
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)

def evaluate_model(
    model,
    dataloader,
    task_type="classification",
    loss_function=None,
    device="cpu",
    logger=None
):
    model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []
    model.to(device)

    with torch.no_grad():
        for features, labels in dataloader:
            features = features.to(device).float()

            if task_type == "classification":
                labels = labels.to(device).long()
            else:
                labels = labels.to(device).float()

            outputs = model(features)

            if isinstance(loss_function, torch.nn.CrossEntropyLoss) and labels.dtype != torch.long:
                labels = labels.long()
            elif not isinstance(loss_function, torch.nn.CrossEntropyLoss) and labels.dtype != torch.float32:
                labels = labels.float()

            if loss_function:
                loss = loss_function(outputs, labels)
                total_loss += loss.item()

            if task_type == "classification":
                preds = torch.argmax(outputs, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
            else:
                all_preds.extend(outputs.cpu().numpy().flatten())
                all_labels.extend(labels.cpu().numpy().flatten())

    avg_loss = total_loss / len(dataloader) if loss_function else None

    if task_type == "classification":
        results = {
            "Loss": avg_loss,
            "Accuracy": accuracy_score(all_labels, all_preds),
            "Precision": precision_score(all_labels, all_preds, average="weighted", zero_division=0),
            "Recall": recall_score(all_labels, all_preds, average="weighted", zero_division=0),
            "F1-Score": f1_score(all_labels, all_preds, average="weighted", zero_division=0),
            "Confusion_Matrix": confusion_matrix(all_labels, all_preds).tolist(),
        }
    else:
        results = {
            "Loss": avg_loss,
            "MSE": mean_squared_error(all_labels, all_preds),
            "MAE": mean_absolute_error(all_labels, all_preds),
            "R²": r2_score(all_labels, all_preds),
        }

    if logger:
        logger.info(f"Evaluation Results: {results}")
    else:
        print(f"Evaluation Results: {results}")

    return results
