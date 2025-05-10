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

        # Stack all images into arrays
        X_train = np.stack(train_df['image'].values)
        X_test = np.stack(test_df['image'].values)

        # For MNIST: reshape to (N, 1, 28, 28) for CNN
        # Add channel dimension and normalize
        if X_train.ndim == 3:  # For MNIST (no channel dimension)
            X_train = X_train[:, np.newaxis, :, :]
            X_test = X_test[:, np.newaxis, :, :]
        
        # Normalize pixel values
        X_train = X_train / 255.0
        X_test = X_test / 255.0

        y_train = train_df['label'].values
        y_test = test_df['label'].values

        return {
            'train': {'X': X_train, 'y': y_train},
            'test': {'X': X_test, 'y': y_test}
        }