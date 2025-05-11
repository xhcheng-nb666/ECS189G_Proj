from local_code.base_class.method import method
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import matplotlib.pyplot as plt
import torch
from torch import nn
import numpy as np


class mnist_CNN(method, nn.Module):
    data = None
    max_epoch = 500
    learning_rate = 1e-3
    batch_size = 256
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # Add device detection

    def __init__(self, mName, mDescription):
        method.__init__(self, mName, mDescription)
        nn.Module.__init__(self)

        # Increase batch size for faster training

        # First convolutional layer: input channels=1, output channels=16
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        # Second convolutional layer: input channels=16, output channels=32
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.3)  # Increased dropout rate
        
        # Increased size of FC layer
        # After two max pooling layers, the image size will be 7x7
        # With 32 channels, the input to FC will be 32 * 7 * 7
        self.fc1 = nn.Linear(32 * 7 * 7, 128)  # Added larger intermediate FC layer
        self.fc2 = nn.Linear(128, 10)  # Output layer
        
        self.relu = nn.ReLU()

        # Move model to GPU
        self.to(self.device)

        # Enable mixed precision training
        self.scaler = torch.amp.GradScaler('cuda')

    def forward(self, x):
        # First conv layer
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        
        # Second conv layer
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        
        # Flatten the tensor for FC layers
        x = x.view(-1, 32 * 7 * 7)
        
        # FC layers with dropout
        x = self.dropout(x)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x

    def evaluate(self, test_y, pred_y):
        return {
            'accuracy': accuracy_score(test_y, pred_y),
            'f1_macro': f1_score(test_y, pred_y, average='macro'),
            'precision_macro': precision_score(test_y, pred_y, average='macro'),
            'recall_macro': recall_score(test_y, pred_y, average='macro')
        }

    def train(self, X, y):
        self.to(self.device)
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=0.01)
        loss_function = nn.CrossEntropyLoss()

        self.loss_history = []
        self.accuracy_history = []

        X_tensor = torch.FloatTensor(X)
        y_tensor = torch.LongTensor(y)

        dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=2
        )

        for epoch in range(self.max_epoch):
            super().train()  # Set model to training mode
            epoch_loss = 0
            epoch_correct = 0
            total_samples = 0

            for batch_X, batch_y in dataloader:
                batch_X = batch_X.to(self.device, non_blocking=True)
                batch_y = batch_y.to(self.device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)

                # Use mixed precision training
                with torch.amp.autocast('cuda'):
                    y_pred = self(batch_X)
                    loss = loss_function(y_pred, batch_y)

                # Scale loss and backward pass
                self.scaler.scale(loss).backward()
                self.scaler.step(optimizer)
                self.scaler.update()

                epoch_loss += loss.item() * len(batch_y)
                epoch_correct += (y_pred.argmax(dim=1) == batch_y).sum().item()
                total_samples += len(batch_y)

            avg_loss = epoch_loss / total_samples
            accuracy = epoch_correct / total_samples

            self.loss_history.append(avg_loss)
            self.accuracy_history.append(accuracy)

            # if epoch % 50 == 0:
            if 1:
                    print(f"Epoch {epoch}, Loss: {avg_loss:.4f}, Acc: {accuracy:.4f}")

    def test(self, X):
        # Save the training mode state
        training = self.training
        # Manually set the model to evaluation mode
        self.training = False

        try:
            with torch.no_grad(), torch.amp.autocast('cuda'):
                X_tensor = torch.FloatTensor(X)
                dataloader = torch.utils.data.DataLoader(
                    X_tensor,
                    batch_size=self.batch_size * 2,
                    shuffle=False,
                    num_workers=2,
                    pin_memory=True,
                    persistent_workers=True
                )

                predictions = []
                for batch_X in dataloader:
                    batch_X = batch_X.to(self.device, non_blocking=True)
                    y_pred = self(batch_X)
                    predictions.append(y_pred.cpu())

                predictions = torch.cat(predictions)
                return predictions.argmax(dim=1).numpy()
        finally:
            # Restore the original training mode state
            self.training = training


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