from local_code.base_class.method import method
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import matplotlib.pyplot as plt
import torch
from torch import nn
import numpy as np


class First_CNN(method, nn.Module):
    data = None
    max_epoch = 500
    learning_rate = 1e-3
    batch_size = 64  # Added batch size parameter

    def __init__(self, mName, mDescription):
        method.__init__(self, mName, mDescription)
        nn.Module.__init__(self)

        # Same architecture with dropout added
        self.conv1 = nn.Conv2d(1, 4, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.2)  # Added dropout layer
        self.fc1 = nn.Linear(4 * 14 * 14, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = x.view(-1, 4 * 14 * 14)
        x = self.dropout(x)  # Apply dropout before the final layer
        x = self.fc1(x)
        return x

    def evaluate(self, test_y, pred_y):
        return {
            'accuracy': accuracy_score(test_y, pred_y),
            'f1_macro': f1_score(test_y, pred_y, average='macro'),
            'precision_macro': precision_score(test_y, pred_y, average='macro'),
            'recall_macro': recall_score(test_y, pred_y, average='macro')
        }

    def train(self, X, y):
        # Switch to AdamW optimizer with small weight decay
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=0.01)
        loss_function = nn.CrossEntropyLoss()

        self.loss_history = []
        self.accuracy_history = []

        # Convert data to tensors
        X_tensor = torch.FloatTensor(X)
        y_tensor = torch.LongTensor(y)

        # Create dataset and dataloader for batch processing
        dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        for epoch in range(self.max_epoch):
            epoch_loss = 0
            epoch_correct = 0
            total_samples = 0

            # Training loop with batches
            for batch_X, batch_y in dataloader:
                y_pred = self.forward(batch_X)
                loss = loss_function(y_pred, batch_y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item() * len(batch_y)
                epoch_correct += (y_pred.argmax(dim=1) == batch_y).sum().item()
                total_samples += len(batch_y)

            # Calculate average loss and accuracy for the epoch
            avg_loss = epoch_loss / total_samples
            accuracy = epoch_correct / total_samples

            self.loss_history.append(avg_loss)
            self.accuracy_history.append(accuracy)

            if epoch % 50 == 0:
                print(f"Epoch {epoch}, Loss: {avg_loss:.4f}, Acc: {accuracy:.4f}")

    def test(self, X):
        with torch.no_grad():
            y_pred = self.forward(torch.FloatTensor(X))
            return y_pred.argmax(dim=1).numpy()

    def plot_learning_curves(self):
        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        plt.plot(self.loss_history)
        plt.title('Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')

        plt.subplot(1, 2, 2)
        plt.plot(self.accuracy_history)
        plt.title('Training Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')

        plt.tight_layout()
        plt.savefig('learning_curves_simple.png')
        print("Saved learning_curves_simple.png")

    def run(self):
        print('method running...')
        print('--start training...')
        self.train(self.data['train']['X'], self.data['train']['y'])
        print('--start testing...')
        pred_y = self.test(self.data['test']['X'])
        true_y = self.data['test']['y']

        print('\n--evaluation metrics--')
        metrics = self.evaluate(true_y, pred_y)
        for k, v in metrics.items():
            print(f'{k}: {v:.4f}')
        print('------------------------\n')

        return metrics

    def save_model(self, path='model_weights.pth'):
        """Save model weights to a file"""
        torch.save(self.state_dict(), path)
        print(f"Model weights saved to {path}")

    def load_model(self, path='model_weights.pth'):
        """Load model weights from a file"""
        self.load_state_dict(torch.load(path))
        self.eval()  # Set the model to evaluation mode
        print(f"Model weights loaded from {path}")