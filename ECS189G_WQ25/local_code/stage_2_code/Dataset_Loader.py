import pickle
import pandas as pd
import numpy as np
from local_code.base_class.dataset import dataset


class Dataset_Loader(dataset):
    def __init__(self, dName, dDescription, datasetPath):
        dataset.__init__(self, dName, dDescription)
        self.data_path = datasetPath
        self.name = dName

    def load(self):
        print(f'loading data for {self.name}...')
        f = open(self.data_path, 'rb')
        data = pickle.load(f)
        f.close()

        train_df = pd.DataFrame(data['train'])
        test_df = pd.DataFrame(data['test'])

        if self.name == "CIFAR":
            # Convert CIFAR data from [N, H, W, C] to [N, C, H, W] format
            X_train = np.stack(train_df['image'].values)
            X_test = np.stack(test_df['image'].values)

            # Transpose the dimensions to PyTorch format
            X_train = X_train.transpose(0, 3, 1, 2)
            X_test = X_test.transpose(0, 3, 1, 2)
        else:
            # For MNIST, add channel dimension
            X_train = np.stack(train_df['image'].values)[:, np.newaxis, :, :]
            X_test = np.stack(test_df['image'].values)[:, np.newaxis, :, :]

        # Normalize pixel values
        X_train = X_train.astype(np.float32) / 255.0
        X_test = X_test.astype(np.float32) / 255.0

        y_train = train_df['label'].values
        y_test = test_df['label'].values

        return {
            'train': {'X': X_train, 'y': y_train},
            'test': {'X': X_test, 'y': y_test}
        }