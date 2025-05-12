import pickle
import numpy as np
import torch
from torch.utils.data import Dataset  # Add this import
from local_code.base_class.dataset import dataset


class Dataset_Loader(dataset):
    def __init__(self, dName, dDescription, datasetPath):
        dataset.__init__(self, dName, dDescription)
        self.data_path = datasetPath
        self.name = dName

    def load(self):
        print(f'loading data for {self.name}...')
        with open(self.data_path, 'rb') as f:
            data = pickle.load(f)

        # Extract training data
        train_data = data['train']
        X_train = np.stack([item['image'] for item in train_data])
        y_train = np.array([item['label'] for item in train_data])

        # Extract test data
        test_data = data['test']
        X_test = np.stack([item['image'] for item in test_data])
        y_test = np.array([item['label'] for item in test_data])

        # For MNIST: reshape to (N, 1, 28, 28) for CNN
        # Add channel dimension and normalize
        if X_train.ndim == 3:  # For MNIST (no channel dimension)
            X_train = X_train[:, np.newaxis, :, :]
            X_test = X_test[:, np.newaxis, :, :]

        # Normalize pixel values
        X_train = X_train / 255.0
        X_test = X_test / 255.0

        return {
            'train': {'X': X_train, 'y': y_train},
            'test': {'X': X_test, 'y': y_test}
        }

class NumpyDataset(Dataset):
    def __init__(self, images, labels):
        # Convert to tensors once during initialization
        self.images = torch.from_numpy(images).float().div(255.0)  # Normalize here
        if len(self.images.shape) == 3:
            self.images = self.images.unsqueeze(1)  # Add channel dimension if needed
        self.labels = torch.from_numpy(labels).long()
        
    def __getitem__(self, index):
        # Now just return tensor slices - much faster
        return self.images[index], self.labels[index]
        
    def __len__(self):
        return len(self.images)