import sys
import os
import pickle
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../ECS189G_WQ25")))
import numpy as np
from local_code.stage_3_code.Dataset_Loader import Dataset_Loader
from local_code.stage_3_code.first_CNN import First_CNN
from local_code.stage_3_code.cifar_CNN import RGB_CNN
from local_code.stage_3_code.mnist_modified import mnist_CNN
from local_code.stage_3_code.simple_cifar import CIFAR_CNN
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
    
    os.makedirs("../../result/stage_3_result", exist_ok=True)
    save_path = os.path.join("../../result/stage_3_result", fig_name)
    plt.savefig(save_path)
    print(f"Saved {save_path}")

def save_metrics(metrics, time, filename):
    os.makedirs("../../result/stage_3_result", exist_ok=True)
    path = os.path.join("../../result/stage_3_result", filename)
    with open(path, 'w') as f:
        for k, v in metrics.items():
            f.write(f"{k}: {v}\n")
        f.write(f"\nTime elapsed: {time:.2f} seconds")
    print(f"Saved {path}")


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
    elapsed_time = end - start
    save_metrics(metrics, elapsed_time, f"{model_name}_performance.txt")
    print(f"Time elapsed to train {model_name}: {end - start:.2f} seconds")

def main():
    # Enable cuDNN autotuner
    torch.backends.cudnn.benchmark = True

    # Initialize multiprocessing safely for Windows
    import torch.multiprocessing as mp
    mp.freeze_support()
    if mp.get_start_method(allow_none=True) is None:
        mp.set_start_method('spawn')

    # run_cnn("Simple CNN for MNIST", First_CNN, "MNIST", MNIST_PATH, "MNIST_learning_curves.png")
    # run_cnn("second MNIST model", mnist_CNN, "MNIST", MNIST_PATH, "second_MNIST_learning_curves.png")
    # run_cnn("firt_CIFAR_CNN", RGB_CNN, "CIFAR", CIFAR_PATH, "CIFAR_learning_curves.png")
    run_cnn("simple_CIFAR", CIFAR_CNN, "CIFAR", CIFAR_PATH, "simple_CIFAR_learning_curves.png")

if __name__ == "__main__":
    main()