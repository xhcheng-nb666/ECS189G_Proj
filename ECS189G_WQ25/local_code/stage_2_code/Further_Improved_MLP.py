'''
Concrete MethodModule class for a specific learning MethodModule
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from local_code.base_class.method import method
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import matplotlib.pyplot as plt
import torch
from torch import nn
import numpy as np


class Method_Further_Improved_MLP(method, nn.Module):
    data = None
    max_epoch = 500
    learning_rate = 1e-3

    def __init__(self, mName, mDescription):
        method.__init__(self, mName, mDescription)
        nn.Module.__init__(self)

        # First hidden layer
        self.fc_layer_1 = nn.Linear(784, 256)
        self.activation_func_1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.2)

        # Second hidden layer
        self.fc_layer_2 = nn.Linear(256, 128)
        self.activation_func_2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.2)

        # Output layer
        self.fc_layer_3 = nn.Linear(128, 10)
        self.activation_func_3 = nn.Softmax(dim=1)

    def forward(self, x):
        '''Forward propagation'''
        # First hidden layer with dropout
        h1 = self.dropout1(self.activation_func_1(self.fc_layer_1(x)))
        # Second hidden layer with dropout
        h2 = self.dropout2(self.activation_func_2(self.fc_layer_2(h1)))
        # Output layer
        y_pred = self.activation_func_3(self.fc_layer_3(h2))
        return y_pred


    # backward error propagation will be implemented by pytorch automatically
    # so we don't need to define the error backpropagation function here

    # stage 2-2 & 2-3
    def train(self, X, y):
        batch_size = 128
        dataset = torch.utils.data.TensorDataset(
            torch.FloatTensor(np.array(X)),
            torch.LongTensor(np.array(y))
        )
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=True
        )

        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=1e-4)

        loss_function = torch.nn.CrossEntropyLoss(label_smoothing=0.1)
        # check here for the torch.optim doc: https://pytorch.org/docs/stable/optim.html
        # optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=1e-4)
        # check here for the nn.CrossEntropyLoss doc: https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html
        # loss_function = torch.nn.CrossEntropyLoss(label_smoothing=0.1)
        # for training accuracy investigation purpose
        # accuracy_evaluator = Evaluate_Accuracy('training evaluator', '')

        self.loss_history = []
        self.accuracy_history = []

        # it will be an iterative gradient updating process
        # we don't do mini-batch, we use the whole input as one batch
        # you can try to split X and y into smaller-sized batches by yourself
        for epoch in range(self.max_epoch): # you can do an early stop if self.max_epoch is too much...
            # get the output, we need to covert X into torch.tensor so pytorch algorithm can operate on it
            # y_pred = self.forward(torch.FloatTensor(np.array(X)))
            epoch_loss = 0.0
            epoch_accuracy = 0.0
            for X_batch, y_batch in dataloader:
                # convert y to torch.tensor as well
                # y_true = torch.LongTensor(np.array(y))
                # calculate the training loss
                y_pred = self.forward(X_batch)
                train_loss = loss_function(y_pred, y_batch)

                # check here for the gradient init doc: https://pytorch.org/docs/stable/generated/torch.optim.Optimizer.zero_grad.html
                optimizer.zero_grad()
                # check here for the loss.backward doc: https://pytorch.org/docs/stable/generated/torch.Tensor.backward.html
                # do the error backpropagation to calculate the gradients
                train_loss.backward()
                # check here for the opti.step doc: https://pytorch.org/docs/stable/optim.html
                # update the variables according to the optimizer and the gradients calculated by the above loss.backward function
                optimizer.step()

                epoch_loss += train_loss.item()
                epoch_accuracy += accuracy_score(y_batch.numpy(), y_pred.argmax(dim=1).numpy())

            epoch_loss /= len(dataloader)
            epoch_accuracy /= len(dataloader)

            self.loss_history.append(epoch_loss)
            self.accuracy_history.append(epoch_accuracy)


            if epoch%100 == 0:
                # pred_labels = y_pred.argmax(dim=1).numpy()
                print(f"Epoch {epoch}, Loss: {epoch_loss:.4f}, Acc: {epoch_accuracy:.4f}")

    def evaluate(self, y_true, y_pred):
        return {
            'accuracy': accuracy_score(y_true, y_pred),
            'f1_macro': f1_score(y_true, y_pred, average='macro'),
            'precision_macro': precision_score(y_true, y_pred, average='macro'),
            'recall_macro': recall_score(y_true, y_pred, average='macro')
        }

    def test(self, X):
        # do the testing, and result the result
        y_pred = self.forward(torch.FloatTensor(np.array(X)))
        # convert the probability distributions to the corresponding labels
        # instances will get the labels corresponding to the largest probability
        return y_pred.max(1)[1]

    def plot_learning_curves(model):
        plt.figure(figsize=(12, 5))

        # Plot Loss
        plt.subplot(1, 2, 1)
        plt.plot(model.loss_history, label='Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Loss Curve with 4 layers')
        plt.legend()

        # Plot Accuracy
        plt.subplot(1, 2, 2)
        plt.plot(model.accuracy_history, label='Training Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Training Accuracy Curve with 4 layers')
        plt.legend()

        plt.tight_layout()
        plt.savefig('learning_curves_update.png')
        print("Saved learning_curves_update.png")

    def run(self):
        print('method running...')
        print('--start training...')
        self.train(self.data['train']['X'], self.data['train']['y'])
        print('--start testing...')
        pred_y = self.test(self.data['test']['X'])
        true_y = self.data['test']['y']

        metrics = self.evaluate(true_y, pred_y.numpy())
        for k, v in metrics.items():
            print(f'{k}: {v:.4f}')

        return metrics