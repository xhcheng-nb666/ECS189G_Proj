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
import numpy as np


class Method_CNN1(method, nn.Module):
    data = None
    max_epoch = 501
    learning_rate = 1e-3

    def __init__(self, mName, mDescription):
        method.__init__(self, mName, mDescription)
        nn.Module.__init__(self)

        self.conv1 = nn.Conv2d(1, 16, kernel_size=5)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5)

        # After conv1 → pool → conv2 → pool:
        # Input: (1, 112, 92)
        # Conv1 (16, 108, 88) → Pool (16, 54, 44)
        # Conv2 (32, 50, 40) → Pool (32, 25, 20)

        self.fc1 = nn.Linear(32 * 25 * 20, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 40)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 32 * 25 * 20)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def train(self, X, y):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate,)
        loss_function = nn.CrossEntropyLoss()

        self.loss_history = []
        self.accuracy_history = []

        X_tensor = torch.FloatTensor(np.array(X)).unsqueeze(1)  # (N, 1, 112, 92)
        y_tensor = torch.LongTensor(np.array(y))

        for epoch in range(self.max_epoch):
            optimizer.zero_grad()
            outputs = self.forward(X_tensor)
            loss = loss_function(outputs, y_tensor)
            loss.backward()
            optimizer.step()

            acc = accuracy_score(y_tensor.numpy(), outputs.argmax(dim=1).numpy())
            self.loss_history.append(loss.item())
            self.accuracy_history.append(acc)

            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss: {loss.item():.4f}, Acc: {self.accuracy_history[-1]:.4f}")


    def evaluate(self, y_true, y_pred):
        return {
            'accuracy': accuracy_score(y_true, y_pred),
            'f1_macro': f1_score(y_true, y_pred, average='macro'),
            'precision_macro': precision_score(y_true, y_pred, average='macro'),
            'recall_macro': recall_score(y_true, y_pred, average='macro')
        }

    def test(self, X):
        X_tensor = torch.FloatTensor(np.array(X)).unsqueeze(1)
        outputs = self.forward(X_tensor)
        return outputs.argmax(dim=1)

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
