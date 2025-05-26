import pickle
import numpy as np
from torchvision import transforms
from PIL import Image

class ORLDatasetLoader:
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path

    def load(self):
        print('loading ORL data...')
        with open(self.dataset_path, 'rb') as f:
            raw_data = pickle.load(f)

        def preprocess(dataset_split):
            X, y = [], []
            for item in dataset_split:
                # Grayscale: use only R channel
                img = np.array(item['image'])[:, :, 0] / 255.0  # normalize
                X.append(img.astype(np.float32))
                y.append(int(item['label']) - 1)  # from 1–40 to 0–39
            return X, y

        X_train, y_train = preprocess(raw_data['train'])
        X_test, y_test = preprocess(raw_data['test'])

        return {
            'train': {'X': X_train, 'y': y_train},
            'test': {'X': X_test, 'y': y_test}
        }
    

class MNISTDatasetLoader:
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path

    def load(self):
        print('loading MNIST data...')

        with open(self.dataset_path, 'rb') as f:
            raw_data = pickle.load(f)

        def preprocess(dataset_split):
            X, y = [], []
            for item in dataset_split:
                img = np.array(item['image']) / 255.0  # shape: (28, 28)
                X.append(img.astype(np.float32))
                y.append(int(item['label']))
            return X, y

        X_train, y_train = preprocess(raw_data['train'])
        X_test, y_test = preprocess(raw_data['test'])

        return {
            'train': {'X': X_train, 'y': y_train},
            'test': {'X': X_test, 'y': y_test}
        }
    
class CIFARDatasetLoader:
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path

    def load(self):
        print('loading CIFAR data...')

        with open(self.dataset_path, 'rb') as f:
            raw_data = pickle.load(f)

        def preprocess(dataset_split):
            X, y = [], []
            augment = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor()  # this gives a (C, H, W) tensor normalized to [0,1]
            ])
            for item in dataset_split:
                img = np.array(item['image'])  # shape: (32, 32, 3)

                # Convert to PIL.Image to use torchvision transforms
                img_pil = Image.fromarray(img.astype(np.uint8))

                # Apply transform to get tensor of shape (3, 32, 32)
                img_tensor = augment(img_pil).numpy()

                X.append(img_tensor.astype(np.float32))
                y.append(int(item['label']))
    
            return X, y

        X_train, y_train = preprocess(raw_data['train'])
        X_test, y_test = preprocess(raw_data['test'])

        return {
            'train': {'X': X_train, 'y': y_train},
            'test': {'X': X_test, 'y': y_test}
        }
