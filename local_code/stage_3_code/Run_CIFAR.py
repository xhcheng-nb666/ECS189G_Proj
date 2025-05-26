import sys
import os

# Add the parent `local_code` directory to the module search path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from Method_CNN import Method_CNN
from Dataset_Loader import CIFARDatasetLoader

def main():
    print("=== Running CNN Model ===")

    dataset_path = os.path.abspath(os.path.join(os.path.dirname(__file__),
                          '../../data/stage_3_data/CIFAR'))

    if not os.path.isfile(dataset_path):
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    dataset_loader = CIFARDatasetLoader(dataset_path)
    data = dataset_loader.load()

    model = Method_CNN('CNN_CIFAR', 'CNN for CIFAR dataset', input_channels=3, num_classes=10)
    model.data = data
    model.run()
    

if __name__ == '__main__':
    main()