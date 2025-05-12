from local_code.base_class.method import method
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import matplotlib.pyplot as plt
import torch
from torch import nn
import numpy as np


class CIFAR_CNN(method, nn.Module):
    data = None
    max_epoch = 500
    learning_rate = 1e-3
    batch_size = 256
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def __init__(self, mName, mDescription):
        method.__init__(self, mName, mDescription)
        nn.Module.__init__(self)

        # Reduced architecture for CIFAR-10
        self.features = nn.Sequential(
            # First block
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.25),

            # Second block
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.25)
        )

        self.classifier = nn.Sequential(
            nn.Linear(64 * 8 * 8, 128),  # Changed from 64*4*4 to 64*8*8 due to one less pooling layer
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, 10)  # Reduced from 256 to 128
        )

        # Initialize weights
        self._initialize_weights()
        self.to(self.device)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        if x.dim() == 4 and x.size(1) != 3:
            x = x.permute(0, 3, 1, 2)
        x = self.features(x)
        x = x.reshape(x.size(0), -1)
        x = self.classifier(x)
        return x

    def evaluate(self, test_y, pred_y):
        return {
            'accuracy': accuracy_score(test_y, pred_y),
            'f1_macro': f1_score(test_y, pred_y, average='macro'),
            'precision_macro': precision_score(test_y, pred_y, average='macro'),
            'recall_macro': recall_score(test_y, pred_y, average='macro')
        }

    def train_model(self, X, y):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5,
                                                               verbose=True)
        loss_function = nn.CrossEntropyLoss()

        self.loss_history = []
        self.accuracy_history = []

        X_tensor = torch.FloatTensor(X).to(self.device)
        if X_tensor.dim() == 4 and X_tensor.size(1) != 3:
            X_tensor = X_tensor.permute(0, 3, 1, 2)
        y_tensor = torch.LongTensor(y).to(self.device)

        dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        for epoch in range(self.max_epoch):
            self.train()  # Set model to training mode
            epoch_loss = 0
            epoch_correct = 0
            total_samples = 0

            for batch_X, batch_y in dataloader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                y_pred = self.forward(batch_X)
                loss = loss_function(y_pred, batch_y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item() * len(batch_y)
                epoch_correct += (y_pred.argmax(dim=1) == batch_y).sum().item()
                total_samples += len(batch_y)

            avg_loss = epoch_loss / total_samples
            accuracy = epoch_correct / total_samples

            self.loss_history.append(avg_loss)
            self.accuracy_history.append(accuracy)

            scheduler.step(avg_loss)

            if epoch % 25 == 0:
                    print(f"Epoch {epoch}, Loss: {avg_loss:.4f}, Acc: {accuracy:.4f}")

    def test(self, X):
        self.eval()  # Set model to evaluation mode
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            if X_tensor.dim() == 4 and X_tensor.size(1) != 3:
                X_tensor = X_tensor.permute(0, 3, 1, 2)
            y_pred = self.forward(X_tensor)
            return y_pred.cpu().argmax(dim=1).numpy()

    def plot_learning_curves(self, fig_name):
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
        plt.savefig(fig_name)
        print(f"Saved {fig_name}")

    def run(self):
        print('method running...')
        print('--start training...')
        self.train_model(self.data['train']['X'], self.data['train']['y'])
        print('--start testing...')
        pred_y = self.test(self.data['test']['X'])
        true_y = self.data['test']['y']

        print('\n--evaluation metrics--')
        metrics = self.evaluate(true_y, pred_y)
        for k, v in metrics.items():
            print(f'{k}: {v:.4f}')
        print('------------------------\n')
        return metrics

    def save_model(self, path='rgb_model_simple.pth'):
        torch.save(self.state_dict(), path)
        print(f"Model weights saved to {path}")

    def load_model(self, path='rgb_model_simple.pth'):
        self.load_state_dict(torch.load(path))
        self.eval()
        print(f"Model weights loaded from {path}")