import os
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import seaborn as sns

from build_vocab import build_vocab
from token_and_augment import augment_dataframe, token_and_augment
from evaluation import evaluate

# Config / Hyperparameters
data_dir = "data"
train_csv = os.path.join(data_dir, "atis_train.csv")
val_csv   = os.path.join(data_dir, "atis_val.csv")
test_csv  = os.path.join(data_dir, "atis_test.csv")

seed = 42
batch_size = 32
max_len = 40
embed_dim = 200
hidden_dim = 128
num_epochs = 20
lr = 1e-3
augment_frac = 1.0
aug_ins = 0.05
aug_del = 0.05
aug_rep = 0.05

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Folders for saving models and plots
os.makedirs("paths", exist_ok=True)
os.makedirs("plots", exist_ok=True)


# Reproducibility
def set_seed(seed=seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
set_seed(seed)


# Dataset Class
class ATISIntentDataset(Dataset):
    def __init__(self, df: pd.DataFrame, word2idx: dict, augment: bool=False, max_len: int=max_len):
        self.df = df.reset_index(drop=True).copy()
        self.word2idx = word2idx
        self.augment = augment
        self.max_len = max_len
        self.df["label"] = self.df["label"].astype(int)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        text = row["text"]
        label = int(row["label"])
        seq = token_and_augment(text, self.word2idx, augment=self.augment, max_len=self.max_len)
        return torch.tensor(seq, dtype=torch.long), torch.tensor(label, dtype=torch.long)


# 2-Layer BiLSTM Without Attention Model
class BiLSTM2Layer(nn.Module):
    """
    Two-layer bidirectional LSTM without attention.
    Concatenates final forward & backward hidden states from the top layer
    and applies FC + dropout.
    """
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=2,       # two layers
            batch_first=True,
            bidirectional=True,
            dropout=0.2         # dropout between LSTM layers
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x):
        emb = self.embedding(x)  # [batch, seq_len, embed_dim]
        _, (h_n, _) = self.lstm(emb)  # h_n: [num_layers*2, batch, hidden_dim]
        # concatenate final forward and backward hidden states from top layer
        h_forward = h_n[-2]   # second layer, forward direction
        h_backward = h_n[-1]  # second layer, backward direction
        h_cat = torch.cat([h_forward, h_backward], dim=1)
        h_cat = self.dropout(h_cat)
        logits = self.fc(h_cat)
        return logits


# Helpers
def compute_class_weights(df):
    counts = df["label"].value_counts().sort_index()
    labels = counts.index.tolist()
    freqs = counts.values.astype(float)
    total = freqs.sum()
    weights = total / (len(freqs) * freqs)
    weight_map = {lab: float(w) for lab, w in zip(labels, weights)}
    max_label = int(df["label"].max())
    weight_list = [weight_map.get(i, 1.0) for i in range(max_label + 1)]
    return torch.tensor(weight_list, dtype=torch.float, device=device)

def compute_accuracy(model, loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for seqs, labels in loader:
            seqs, labels = seqs.to(device), labels.to(device)
            preds = model(seqs).argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return 100.0 * correct / total if total > 0 else 0.0


# Main Training Function
def main():
    print("Loading CSVs...")
    train_df = pd.read_csv(train_csv)
    val_df   = pd.read_csv(val_csv)
    test_df  = pd.read_csv(test_csv)

    # Augment training set
    print(f"Original train size: {len(train_df)}. Augmenting frac={augment_frac} ...")
    aug_train_df = augment_dataframe(
        train_df,
        frac=augment_frac,
        ins_p=aug_ins,
        del_p=aug_del,
        rep_p=aug_rep
    )
    print(f"Augmented train size: {len(aug_train_df)}")

    # Build vocabulary
    word2idx, idx2word = build_vocab(train_csv, min_freq=1, save_json=False)

    # DataLoaders
    train_ds = ATISIntentDataset(aug_train_df, word2idx, augment=True, max_len=max_len)
    val_ds   = ATISIntentDataset(val_df, word2idx, augment=False, max_len=max_len)
    test_ds  = ATISIntentDataset(test_df, word2idx, augment=False, max_len=max_len)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    # Model setup
    max_label = int(max(train_df["label"].max(), val_df["label"].max(), test_df["label"].max()))
    num_classes = max_label + 1
    vocab_size = len(word2idx)

    model = BiLSTM2Layer(vocab_size, embed_dim, hidden_dim, num_classes).to(device)
    class_weights = compute_class_weights(aug_train_df)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Training loop
    train_losses, test_losses = [], []
    train_accs, test_accs = [], []

    best_val = 0.0
    model_name = "BiLSTM2Layer"
    best_path = os.path.join("paths", f"{model_name}.pt")

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}...")
        model.train()
        running_loss = 0.0

        for inputs, labels in tqdm(train_loader, desc="train"):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        train_loss = running_loss / len(train_loader)
        train_acc = compute_accuracy(model, train_loader)
        test_acc = compute_accuracy(model, test_loader)

        # Validation loss
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                val_loss += criterion(outputs, labels).item()
        test_loss = val_loss / len(test_loader)

        # Save metrics
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        train_accs.append(train_acc)
        test_accs.append(test_acc)

        print(f"Train Loss: {train_loss:.4f} | Test Loss: {test_loss:.4f} | "
              f"Train Acc: {train_acc:.2f}% | Test Acc: {test_acc:.2f}%")

        # Save best model
        if test_acc > best_val:
            best_val = test_acc
            torch.save({
                "model_state": model.state_dict(),
                "word2idx": word2idx,
                "idx2word": idx2word
            }, best_path)
            print(f"Saved best model (test_acc={best_val:.2f}%) -> {best_path}")

    print("\nTraining complete\n")


    # Plot metrics
    epochs = list(range(1, num_epochs+1))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14,5))

    ax1.plot(epochs, train_losses, 'b-', label='Training Loss')
    ax1.plot(epochs, test_losses, 'r-', label='Test Loss')
    ax1.set_title(f'{model_name} Loss Curves')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Cross-Entropy Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, train_accs, 'b-', label='Training Accuracy')
    ax2.plot(epochs, test_accs, 'r-', label='Test Accuracy')
    ax2.set_title(f'{model_name} Accuracy Curves')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join("plots", f"{model_name}_loss_accuracy_curves.png"))
    plt.close(fig)


    # Final Evaluation
    print("\nLoading best saved model for final evaluation...")
    checkpoint = torch.load(best_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])

    label_id_to_string = (
        train_df.sort_values("label")
                .drop_duplicates("label")[["label","intent"]]
                .set_index("label")["intent"]
                .to_dict()
    )

    metrics, y_true, y_pred, cm = evaluate(
        model,
        test_loader,
        label_id_to_string,
        device=device,
        plot_cm=False
    )

    # Only use labels present in test set for confusion matrix
    test_labels_present = sorted(set(y_true))
    target_names = [label_id_to_string[i] for i in test_labels_present]

    # Save confusion matrix plot
    plt.figure(figsize=(10,8))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=target_names,
        yticklabels=target_names
    )
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.title(f"{model_name} Confusion Matrix")
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join("plots", f"{model_name}_confusion_matrix.png"))
    plt.close()

    # Print metrics
    print("\nFINAL TEST METRICS")
    print("Accuracy:", metrics["accuracy"])
    print(f"Macro Precision: {metrics['precision_macro']:.4f}")
    print(f"Macro Recall:    {metrics['recall_macro']:.4f}")
    print(f"Macro F1:        {metrics['f1_macro']:.4f}")
    print("\nWeighted Precision:", metrics["precision_weighted"])
    print("Weighted Recall:   ", metrics["recall_weighted"])
    print("Weighted F1:       ", metrics["f1_weighted"])
    print("\nClassification Report:")
    print(metrics["classification_report"])

if __name__ == "__main__":
    main()