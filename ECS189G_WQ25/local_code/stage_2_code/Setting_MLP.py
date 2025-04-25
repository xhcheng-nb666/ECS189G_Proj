import sys
import os

# Add the correct parent directory of 'local_code' to sys.path
# modify this if needed
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../ECS189G_WQ25")))

from local_code.stage_2_code.Dataset_Loader import Dataset_Loader
from local_code.stage_2_code.Method_MLP import Method_MLP
from local_code.stage_2_code.Improved_MLP import Method_Improved_MLP

# edit this path for different people
dataset = Dataset_Loader("digit_recognition", "", "/home/xhangc/ss1_ECS/ECS189G/ECS189G_WQ25/data/stage_2_data/train.csv", "/home/xhangc/ss1_ECS/ECS189G/ECS189G_WQ25/data/stage_2_data/test.csv")
data = dataset.load()

# BEFORE (original version: Adam + plain CrossEntropyLoss)
print("=== Running Original Model ===")
original_mlp = Method_MLP("mlp_original", "")
original_mlp.data = dataset.load()
original_metrics = original_mlp.run()

# AFTER (new version: AdamW + Label Smoothing)
print("\n=== Running Improved Model (AdamW + Label Smoothing) ===")
improved_mlp = Method_Improved_MLP("mlp_improved", "")
improved_mlp.data = dataset.load()
# Important: modify your Method_MLP to use AdamW + LabelSmoothingCrossEntropy BEFORE running this line
improved_metrics = improved_mlp.run()

import matplotlib.pyplot as plt

def plot_learning_curves_comparison(model1, label1, model2, label2):
    plt.figure(figsize=(14, 6))

    # Plot Loss Comparison
    plt.subplot(1, 2, 1)
    plt.plot(model1.loss_history, label=f'{label1} Loss')
    plt.plot(model2.loss_history, label=f'{label2} Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Comparison')
    plt.legend()

    # Plot Accuracy Comparison
    plt.subplot(1, 2, 2)
    plt.plot(model1.accuracy_history, label=f'{label1} Accuracy')
    plt.plot(model2.accuracy_history, label=f'{label2} Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training Accuracy Comparison')
    plt.legend()

    plt.tight_layout()
    plt.savefig('learning_curves_comparison.png')
    print("Saved learning_curves_compariosn.png")

plot_learning_curves_comparison(original_mlp, "Original", improved_mlp, "Improved")
