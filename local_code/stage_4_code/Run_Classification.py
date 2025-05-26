import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from utils import load_data_from_folders, build_vocab, TextDataset
from model import RNNClassifier
from train import train_model, plot_learning_curves

print("=== Running RNN Text Classification Model ===")

# Step 1: Define paths
print("[INFO] Setting up data paths...")
base_data_path = "/mnt/d/ECS暑课/189G/stage_4_data/text_classification"
train_path = os.path.join(base_data_path, 'train')
test_path = os.path.join(base_data_path, 'test')

# Step 2: Load data
print("[INFO] Loading and processing dataset...")
print("train_path =", train_path)
print("Contents of train_path:", os.listdir(train_path))
for sentiment in ["neg", "pos"]:
    subfolder = os.path.join(train_path, sentiment)
    if not os.path.exists(subfolder):
        print(f"Missing: {subfolder}")
        continue
    files = os.listdir(subfolder)
    print(f"{sentiment}/ contains {len(files)} files")
    print("Example file:", files[0] if files else "(none)")


train_texts, train_labels = load_data_from_folders(train_path)
test_texts, test_labels = load_data_from_folders(test_path)

# Step 3: Build vocabulary
print("[INFO] Building vocabulary...")
vocab = build_vocab(train_texts)
pad_idx = vocab['<pad>']

# Step 4: Create datasets
print("[INFO] Preparing datasets...")
train_dataset = TextDataset(train_texts, train_labels, vocab)
test_dataset = TextDataset(test_texts, test_labels, vocab)

# Step 5: Split into train/val
print("[INFO] Splitting dataset into training and validation sets...")
train_size = int(0.8 * len(train_dataset))
val_size = len(train_dataset) - train_size
train_ds, val_ds = random_split(train_dataset, [train_size, val_size])
print(f"Loaded {len(train_texts)} training examples")
print(f"Loaded {len(test_texts)} testing examples")


# Step 6: Dataloaders
print("[INFO] Creating DataLoaders...")
train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=32)
test_loader = DataLoader(test_dataset, batch_size=32)

# Step 7: Model setup
print("[INFO] Initializing RNN model...")
model = RNNClassifier(vocab_size=len(vocab), embedding_dim=100, hidden_dim=64, output_dim=1, pad_idx=pad_idx)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.BCEWithLogitsLoss()

# Step 8: Train model
print("[INFO] Starting training...")
train_loss, val_loss = train_model(model, train_loader, val_loader, optimizer, criterion, n_epochs=30)
plot_learning_curves(train_loss, val_loss)

# Step 9: Evaluate model
print("[INFO] Evaluating on test set...")
model.eval()
all_preds = []
all_labels = []
with torch.no_grad():
    for texts, labels in test_loader:
        outputs = model(texts)
        preds = (torch.sigmoid(outputs.squeeze()) > 0.5).long()
        all_preds.extend(preds.tolist())
        all_labels.extend(labels.tolist())

correct = sum([p == l for p, l in zip(all_preds, all_labels)])
total = len(all_preds)
accuracy = 100 * correct / total
print(f"[RESULT] Test Accuracy: {accuracy:.4f}%")

# Step 10: Additional metrics
from sklearn.metrics import precision_score, recall_score, f1_score

precision = precision_score(all_labels, all_preds)
recall = recall_score(all_labels, all_preds)
f1 = f1_score(all_labels, all_preds)

print(f"[RESULT] Precision: {precision:.4f}")
print(f"[RESULT] Recall:    {recall:.4f}")
print(f"[RESULT] F1 Score:  {f1:.4f}")
