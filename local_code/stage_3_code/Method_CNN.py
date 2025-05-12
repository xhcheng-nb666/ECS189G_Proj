'''
Concrete MethodModule class for a specific learning MethodModule
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD
import os
import sys
import numpy as np

base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'base_class'))
sys.path.append(base_path)

from method import method
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import matplotlib.pyplot as plt
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


class Method_CNN(method, nn.Module):
    data = None
    max_epoch = 501
    learning_rate = 1e-3

    def __init__(self, mName, mDescription, input_channels=1, num_classes=40):
        method.__init__(self, mName, mDescription)
        nn.Module.__init__(self)

        self.input_channels = input_channels
        self.num_classes = num_classes

        self.conv1 = nn.Conv2d(self.input_channels, 16, kernel_size=5)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)


        self.bn1 = nn.BatchNorm2d(16)
        self.bn2 = nn.BatchNorm2d(32)
        self.bn3 = nn.BatchNorm2d(64)
        


        self.fc1 = None  # Delay init until forward
        self.fc2 = None
        self.fc3 = nn.Linear(128, self.num_classes)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))


        if self.fc1 is None:
            flattened_size = x.shape[1] * x.shape[2] * x.shape[3]
            self.fc1 = nn.Linear(flattened_size, 256).to(x.device)
            self.fc2 = nn.Linear(256, 128).to(x.device)

        x = x.reshape(x.size(0), -1)  # preserve batch size
        x = self.dropout(F.relu(self.fc1(x)))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def train(self, X, y):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        loss_function = nn.CrossEntropyLoss()

        self.loss_history = []
        self.accuracy_history = []

        X_tensor = torch.FloatTensor(np.array(X))
        if X_tensor.ndim == 3:  # grayscale (N, H, W)
            X_tensor = X_tensor.unsqueeze(1)
        elif X_tensor.shape[-1] == 3:  # color image (N, H, W, C)
            X_tensor = X_tensor.permute(0, 3, 1, 2)

        y_tensor = torch.LongTensor(np.array(y))

        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=128, shuffle=True)

        for epoch in range(self.max_epoch):
            epoch_loss = 0.0
            epoch_preds = []
            epoch_labels = []

            for batch_X, batch_y in loader:
                optimizer.zero_grad()
                outputs = self.forward(batch_X)
                loss = loss_function(outputs, batch_y)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item() * batch_X.size(0)
                epoch_preds.extend(outputs.argmax(dim=1).numpy())
                epoch_labels.extend(batch_y.numpy())

            avg_loss = epoch_loss / len(dataset)
            acc = accuracy_score(epoch_labels, epoch_preds)
            self.loss_history.append(avg_loss)
            self.accuracy_history.append(acc)

            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss: {avg_loss:.4f}, Acc: {acc:.4f}")


    def test(self, X):
        X_tensor = torch.FloatTensor(np.array(X))
        if X_tensor.ndim == 3:
            X_tensor = X_tensor.unsqueeze(1)
        elif X_tensor.shape[-1] == 3:
            X_tensor = X_tensor.permute(0, 3, 1, 2)

        outputs = self.forward(X_tensor)
        return outputs.argmax(dim=1)

    def evaluate(self, y_true, y_pred):
        return {
            'accuracy': accuracy_score(y_true, y_pred),
            'f1_macro': f1_score(y_true, y_pred, average='macro', zero_division=0),
            'precision_macro': precision_score(y_true, y_pred, average='macro', zero_division=0),
            'recall_macro': recall_score(y_true, y_pred, average='macro', zero_division=0)
        }

    def plot_learning_curves(self):
        epochs = list(range(len(self.loss_history)))

        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(epochs, self.loss_history, label='Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Loss Curve')
        plt.grid(True)
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(epochs, self.accuracy_history, label='Training Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Training Accuracy Curve')
        plt.grid(True)
        plt.legend()

        plt.tight_layout()
        plt.savefig('learning_curves.png')
        print("Saved learning_curves.png")
        plt.show(block=True)

    def run(self):
        print('method running...')
        print('--start training...')
        self.train(self.data['train']['X'], self.data['train']['y'])

        print('--start testing...')
        pred_y = self.test(self.data['test']['X'])
        true_y = self.data['test']['y']

        print('--evaluation metrics--')
        metrics = self.evaluate(true_y, pred_y.numpy())
        for k, v in metrics.items():
            print(f'{k}: {v:.4f}')

        print('--plotting learning curves--')
        self.plot_learning_curves()
