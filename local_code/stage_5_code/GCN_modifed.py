import torch
import torch.nn as nn
import torch.nn.functional as F
from local_code.base_class.method import method
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp


class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features):
        super(GraphConvolution, self).__init__()
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        self.reset_parameters()

    def reset_parameters(self):
        # Glorot initialization
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x, adj):
        support = torch.mm(x, self.weight)
        output = torch.spmm(adj, support)
        return output


class Method_GCN(method, nn.Module):
    data = None
    max_epoch = 200  # Set to 200 as specified
    learning_rate = 0.01  # As specified
    weight_decay = 5e-4  # L2 regularization as specified
    dropout_rate = 0.3
    hidden_units = 16
    patience = 10

    def __init__(self, mName, mDescription):
        method.__init__(self, mName, mDescription)
        nn.Module.__init__(self)

        self.gc1 = None
        self.gc2 = None
        self.dropout_layer = nn.Dropout(self.dropout_rate)

    def initialize_model(self, input_dim, hidden_dim, output_dim):
        self.gc1 = GraphConvolution(input_dim, hidden_dim)
        self.gc2 = GraphConvolution(hidden_dim, output_dim)

    def forward(self, x, adj):
        # Layer 1: Input → Hidden (ReLU activation)
        x = F.relu(self.gc1(x, adj))
        x = self.dropout_layer(x)
        # Layer 2: Hidden → Output (Softmax activation)
        x = self.gc2(x, adj)
        return F.softmax(x, dim=1)

    def train_model(self, features, adj, labels, idx_train, idx_val):
        # Adam optimizer with specified learning rate and L2 regularization
        optimizer = torch.optim.AdamW(self.parameters(),
                                     lr=self.learning_rate,
                                     weight_decay=self.weight_decay)

        self.loss_history = []
        self.val_loss_history = []
        self.accuracy_history = []
        self.val_accuracy_history = []

        best_val_loss = float('inf')
        patience_counter = 0
        best_weights = None

        for epoch in range(self.max_epoch):
            # Training
            self.train()
            optimizer.zero_grad()
            output = self(features, adj)
            loss_train = F.cross_entropy(output[idx_train], labels[idx_train])
            acc_train = accuracy_score(labels[idx_train].cpu().numpy(),
                                       output[idx_train].argmax(dim=1).cpu().numpy())
            loss_train.backward()
            optimizer.step()

            # Validation
            self.eval()
            with torch.no_grad():
                output = self(features, adj)
                loss_val = F.cross_entropy(output[idx_val], labels[idx_val])
                acc_val = accuracy_score(labels[idx_val].cpu().numpy(),
                                         output[idx_val].argmax(dim=1).cpu().numpy())

            # Store metrics
            self.loss_history.append(loss_train.item())
            self.val_loss_history.append(loss_val.item())
            self.accuracy_history.append(acc_train)
            self.val_accuracy_history.append(acc_val)

            # Early stopping
            if loss_val < best_val_loss:
                best_val_loss = loss_val
                patience_counter = 0
                best_weights = {key: value.cpu().clone() for key, value in self.state_dict().items()}
            else:
                patience_counter += 1

            if patience_counter >= self.patience:
                print(f'Early stopping at epoch {epoch}')
                self.load_state_dict(best_weights)
                break

            if epoch % 10 == 0:
                # if False:
                print(f'Epoch: {epoch:04d}, '
                      f'Train Loss: {loss_train.item():.4f}, '
                      f'Train Acc: {acc_train:.4f}, '
                      f'Val Loss: {loss_val.item():.4f}, '
                      f'Val Acc: {acc_val:.4f}')

    def test(self, features, adj, idx_test):
        self.eval()
        with torch.no_grad():
            output = self(features, adj)
            return output[idx_test].argmax(dim=1)

    def evaluate(self, y_true, y_pred):
        return {
            'accuracy': accuracy_score(y_true, y_pred),
            'f1_macro': f1_score(y_true, y_pred, average='macro'),
            'precision_macro': precision_score(y_true, y_pred, average='macro'),
            'recall_macro': recall_score(y_true, y_pred, average='macro')
        }

    def plot_learning_curves(self, dataset_name=None):
        """Plot and save learning curves with dataset-specific naming"""
        plt.figure(figsize=(12, 5))

        # Loss plot
        plt.subplot(1, 2, 1)
        plt.plot(self.loss_history, label='Train Loss')
        plt.plot(self.val_loss_history, label='Validation Loss')
        plt.title(f'Learning Curves - Loss ({dataset_name})' if dataset_name else 'Learning Curves - Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)

        # Accuracy plot
        plt.subplot(1, 2, 2)
        plt.plot(self.accuracy_history, label='Train Accuracy')
        plt.plot(self.val_accuracy_history, label='Validation Accuracy')
        plt.title(f'Learning Curves - Accuracy ({dataset_name})' if dataset_name else 'Learning Curves - Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True)

        plt.tight_layout()

        # Save with dataset-specific name
        filename = f'gcn_learning_curves_{dataset_name.lower()}.png' if dataset_name else 'gcn_learning_curves.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()

    def run(self):
        print('method running...')
        print('--start training...')

        # Get data
        features = self.data['graph']['X']
        adj = self.data['graph']['utility']['A']
        labels = self.data['graph']['y']
        idx_train = self.data['train_test_val']['idx_train']
        idx_val = self.data['train_test_val']['idx_val']
        idx_test = self.data['train_test_val']['idx_test']

        # Initialize model with specified hidden units
        input_dim = features.shape[1]
        output_dim = len(torch.unique(labels))
        self.initialize_model(input_dim, self.hidden_units, output_dim)

        # Train
        self.train_model(features, adj, labels, idx_train, idx_val)

        print('--start testing...')
        pred_y = self.test(features, adj, idx_test)
        true_y = labels[idx_test]

        print('--evaluation metrics--')
        metrics = self.evaluate(true_y.cpu().numpy(), pred_y.cpu().numpy())
        for k, v in metrics.items():
            print(f'{k}: {v:.4f}')

        print('--plotting learning curves--')
        self.plot_learning_curves()

        return metrics
class Method_GCN_Modified(method, nn.Module):
    data = None
    max_epoch = 300
    learning_rate = 0.005
    weight_decay = 1e-4
    dropout_rate = 0.3
    hidden_units = [64, 32, 16]  # Added another hidden layer with 16 units
    patience = 20
    
    def __init__(self, mName, mDescription):
        method.__init__(self, mName, mDescription)
        nn.Module.__init__(self)
        
        self.gc1 = None
        self.gc2 = None
        self.gc3 = None
        self.gc4 = None  # Added new layer
        self.dropout_layer = nn.Dropout(self.dropout_rate)
        self.batch_norm1 = None
        self.batch_norm2 = None
        self.batch_norm3 = None  # Added new batch norm
        
    def initialize_model(self, input_dim, hidden_dims, output_dim):
        self.gc1 = GraphConvolution(input_dim, hidden_dims[0])
        self.gc2 = GraphConvolution(hidden_dims[0], hidden_dims[1])
        self.gc3 = GraphConvolution(hidden_dims[1], hidden_dims[2])  # Modified
        self.gc4 = GraphConvolution(hidden_dims[2], output_dim)  # Added new layer
        
        self.batch_norm1 = nn.BatchNorm1d(hidden_dims[0])
        self.batch_norm2 = nn.BatchNorm1d(hidden_dims[1])
        self.batch_norm3 = nn.BatchNorm1d(hidden_dims[2])  # Added new batch norm
        
    def forward(self, x, adj):
        # First layer
        x = self.gc1(x, adj)
        x = self.batch_norm1(x)
        x = F.elu(x)
        x = self.dropout_layer(x)
        
        # Second layer
        x = self.gc2(x, adj)
        x = self.batch_norm2(x)
        x = F.elu(x)
        x = self.dropout_layer(x)
        
        # Third layer (new)
        x = self.gc3(x, adj)
        x = self.batch_norm3(x)
        x = F.elu(x)
        x = self.dropout_layer(x)
        
        # Output layer
        x = self.gc4(x, adj)
        return F.log_softmax(x, dim=1)