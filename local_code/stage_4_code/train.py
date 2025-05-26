import torch
from torch import nn
import matplotlib.pyplot as plt

def train_model(model, train_loader, val_loader, optimizer, criterion, n_epochs=30):
    train_loss, val_loss = [], []

    for epoch in range(n_epochs):
        model.train()
        running_loss = 0.0

        for texts, labels in train_loader:
            optimizer.zero_grad()

            outputs = model(texts)  # output is raw logits
            labels = labels.float()  # ensure float dtype

            # Squeeze only if shape mismatch (e.g., [B, 1] vs [B])
            if outputs.shape != labels.shape:
                outputs = outputs.squeeze()

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        train_loss.append(running_loss / len(train_loader))

        model.eval()
        running_val_loss = 0.0

        with torch.no_grad():
            for texts, labels in val_loader:
                outputs = model(texts)
                labels = labels.float()
                if outputs.shape != labels.shape:
                    outputs = outputs.squeeze()
                loss = criterion(outputs, labels)
                running_val_loss += loss.item()

        val_loss.append(running_val_loss / len(val_loader))

        print(f"Epoch {epoch+1}, Train Loss: {train_loss[-1]:.4f}, Val Loss: {val_loss[-1]:.4f}")

    return train_loss, val_loss

def train_generator(model, dataloader, optimizer, criterion, n_epochs=10):
    model.train()
    for epoch in range(n_epochs):
        total_loss = 0
        for inputs, targets in dataloader:
            optimizer.zero_grad()
            outputs, _ = model(inputs)
            outputs = outputs.reshape(-1, outputs.shape[-1])
            targets = targets.reshape(-1)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}, Generator Loss: {avg_loss:.4f}")



def plot_learning_curves(train_loss, val_loss):
    plt.plot(train_loss, label='Train Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Learning Curves")
    plt.savefig('learning_curves-1.png')
    print("Saved learning_curves.png")
    plt.show(block=True)
