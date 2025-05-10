import sys
import os
import pickle
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../ECS189G_WQ25")))
import numpy as np
from local_code.stage_3_code.Dataset_Loader import Dataset_Loader
from local_code.stage_3_code.first_CNN import First_CNN
import time

DATA_PATH ="/Users/raj/Desktop/ECS189G/ECS189G_Proj/ECS189G_WQ25/data/stage_3_data/MNIST"
dataset = Dataset_Loader("MNIST", "", DATA_PATH)
mnist_data = dataset.load()


start = time.time()
# BEFORE (original version: AdamW + plain Label Smoothing)
print("=== Running Basic CNN ===")
cnn = First_CNN("CNN", "")
cnn.data = mnist_data
original_metrics = cnn.run()
cnn.plot_learning_curves()
cnn.save_model('first.pth')

mid = time.time()

print(f"Time elapsed to train first model: {mid - start:.2f} seconds")
