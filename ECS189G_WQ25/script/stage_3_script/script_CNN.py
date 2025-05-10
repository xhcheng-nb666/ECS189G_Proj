import sys
import os
import pickle
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../ECS189G_WQ25")))
import numpy as np
from local_code.stage_3_code.Dataset_Loader import Dataset_Loader
from local_code.stage_3_code.first_CNN import First_CNN
from local_code.stage_3_code.cifar_CNN import RGB_CNN
import time
import matplotlib.pyplot as plt
import torch.multiprocessing as mp
import torch

# MAC
# MNIST_PATH ="/Users/raj/Desktop/ECS189G/ECS189G_Proj/ECS189G_WQ25/data/stage_3_data/MNIST"
# CIFAR_PATH ="/Users/raj/Desktop/ECS189G/ECS189G_Proj/ECS189G_WQ25/data/stage_3_data/CIFAR"

# WINDOWS
MNIST_PATH = "C:/Users/Raj/PycharmProjects/ECS189G_Proj/ECS189G_WQ25/data/stage_3_data/MNIST"
CIFAR_PATH = "C:/Users/Raj/PycharmProjects/ECS189G_Proj/ECS189G_WQ25/data/stage_3_data/CIFAR"


def plot_learning_curves(model, fig_name):
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(model.loss_history)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    plt.subplot(1, 2, 2)
    plt.plot(model.accuracy_history)
    plt.title('Training Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')

    plt.tight_layout()
    plt.savefig(fig_name)
    print(f"Saved {fig_name}")


def run_cnn(model_name, model, dataset_name, dataset_path, plt_name):
    start = time.time()
    dataset = Dataset_Loader(dataset_name, "", dataset_path)
    data = dataset.load()
    cnn = model(model_name, "")
    cnn.data = data
    print(f"=== Running {model_name} ===")
    metrics = cnn.run()
    plot_learning_curves(cnn, plt_name)
    cnn.save_model(model_name + '.pth')
    end = time.time()
    print(f"Time elapsed to train {model_name}: {end - start:.2f} seconds")

def main():
    # Enable cuDNN autotuner
    torch.backends.cudnn.benchmark = True

    # Initialize multiprocessing safely for Windows
    import torch.multiprocessing as mp
    mp.freeze_support()
    if mp.get_start_method(allow_none=True) is None:
        mp.set_start_method('spawn')

    run_cnn("Simple CNN for MNIST", First_CNN, "MNIST", MNIST_PATH, "MNIST_learning_curves.png")


if __name__ == "__main__":
    main()
