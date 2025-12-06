import torch
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

@torch.no_grad()
def evaluate(model, data_loader, label_encoder, device="cpu", plot_cm=False):
    """
    Evaluate a trained multi-class classification model.

    Computes:
        - overall accuracy
        - macro/weighted precision, recall, F1
        - full per-class classification report
        - confusion matrix (optionally visualized)

    Args:
        model: PyTorch model
        data_loader: DataLoader providing (x, labels)
        label_encoder: dict mapping label index → intent string
        device: "cpu" or "cuda"
        plot_cm: if True, plots confusion matrix as heatmap

    Returns:
        metrics: dict with all metrics
        y_true, y_pred: raw integer labels
        cm: confusion matrix (numpy array)
    """
    model.eval()
    all_preds = []
    all_labels = []

    for x, labels in data_loader:
        x = x.to(device)
        labels = labels.to(device)

        logits = model(x)
        preds = torch.argmax(logits, dim=1)

        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

    # Metrics
    acc = accuracy_score(all_labels, all_preds)
    precision_macro   = precision_score(all_labels, all_preds, average='macro')
    recall_macro      = recall_score(all_labels, all_preds, average='macro')
    f1_macro          = f1_score(all_labels, all_preds, average='macro')
    precision_weighted = precision_score(all_labels, all_preds, average='weighted')
    recall_weighted    = recall_score(all_labels, all_preds, average='weighted')
    f1_weighted        = f1_score(all_labels, all_preds, average='weighted')

    # Fix target_names / labels mismatch
    target_labels = sorted(set(all_labels))   # numeric labels actually in test set
    target_names = [label_encoder[i] for i in target_labels]

    cls_report = classification_report(
        all_labels,
        all_preds,
        labels=target_labels,
        target_names=target_names,
        zero_division=0
    )

    cm = confusion_matrix(all_labels, all_preds)


    # Confusion Matrix plot
    if plot_cm:
        plt.figure(figsize=(10,8))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=target_names, yticklabels=target_names)
        plt.ylabel("True Label")
        plt.xlabel("Predicted Label")
        plt.title("Confusion Matrix")
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.show()

    metrics = {
        "accuracy": acc,
        "precision_macro": precision_macro,
        "recall_macro": recall_macro,
        "f1_macro": f1_macro,
        "precision_weighted": precision_weighted,
        "recall_weighted": recall_weighted,
        "f1_weighted": f1_weighted,
        "classification_report": cls_report
    }

    return metrics, all_labels, all_preds, cm
