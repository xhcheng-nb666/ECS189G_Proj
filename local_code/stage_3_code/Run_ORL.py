import sys
import os

# Add the parent `local_code` directory to the module search path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from Method_CNN1 import Method_CNN1
from Dataset_Loader import ORLDatasetLoader

def main():
    print("=== Running CNN Model ===")

    dataset_path = os.path.abspath(os.path.join(os.path.dirname(__file__),
                          '../../data/stage_3_data/ORL'))

    if not os.path.isfile(dataset_path):
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    dataset_loader = ORLDatasetLoader(dataset_path)
    data = dataset_loader.load()

    model = Method_CNN1('CNN_ORL', 'Convolutional Neural Network for ORL')
    #model = Method_CNN1('CNN_ORL', 'Convolutional Neural Network for ORL', input_channels=1, num_classes=40)
    model.data = data
    model.run()

if __name__ == '__main__':
    main()
