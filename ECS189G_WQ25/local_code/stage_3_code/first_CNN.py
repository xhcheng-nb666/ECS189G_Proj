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

    def __init__(self, mName, mDescription):
        method.__init__(self, mName, mDescription)
        nn.Module.__init__(self)

        # Simple architecture
        self.conv1 = nn.Conv2d(1, 4, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(4 * 14 * 14, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = x.view(-1, 4 * 14 * 14)
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
        optimizer = torch.optim.Adam(self.parameters())
        loss_function = nn.CrossEntropyLoss()

        self.loss_history = []
        self.accuracy_history = []

        X_tensor = torch.FloatTensor(X)
        y_tensor = torch.LongTensor(y)

        for epoch in range(self.max_epoch):
            y_pred = self.forward(X_tensor)
            loss = loss_function(y_pred, y_tensor)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            accuracy = accuracy_score(y, y_pred.argmax(dim=1).numpy())

            self.loss_history.append(loss.item())
            self.accuracy_history.append(accuracy)

            if epoch % 50 == 0:
                print(f"Epoch {epoch}, Loss: {loss.item():.4f}, Acc: {accuracy:.4f}")

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