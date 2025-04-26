import sys
import os
import matplotlib.pyplot as plt
import time

# Add the correct parent directory of 'local_code' to sys.path
# modify this if needed
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../ECS189G_WQ25")))

from local_code.stage_2_code.Dataset_Loader import Dataset_Loader
from local_code.stage_2_code.Method_MLP import Method_MLP
from local_code.stage_2_code.Improved_MLP import Method_Improved_MLP
from local_code.stage_2_code.Further_Improved_MLP import Method_Further_Improved_MLP

# /Users/raj/Desktop/ECS189G/ECS189G_Proj/ECS189G_WQ25/data/stage_2_data/train.csv
# /Users/raj/Desktop/ECS189G/ECS189G_Proj/ECS189G_WQ25/data/stage_2_data/test.csv

# edit this path for different people

# set as needed
DATA_TRAIN_PATH = "/Users/raj/Desktop/ECS189G/ECS189G_Proj/ECS189G_WQ25/data/stage_2_data/train.csv"
DATA_TEST_PATH = "/Users/raj/Desktop/ECS189G/ECS189G_Proj/ECS189G_WQ25/data/stage_2_data/test.csv"

dataset = Dataset_Loader("digit_recognition", "",DATA_TRAIN_PATH, DATA_TEST_PATH)
data = dataset.load()


start = time.time()
# BEFORE (original version: AdamW + plain Label Smoothing)
print("=== Running 3 Layer Model with AdamW and Label Smoothing ===")
original_mlp = Method_Improved_MLP("mlp_improved (3 layers)", "")
original_mlp.data = dataset.load()
original_metrics = original_mlp.run()
original_mlp.plot_learning_curves()

mid = time.time()

print(f"Time elapsed to train first model: {mid - start:.2f} seconds")


# AFTER (new version: AdamW + Label Smoothing + 4 Layers + Dropout)
print("\n=== Running best Model (4 Layers and drop out) ===")
improved_mlp = Method_Further_Improved_MLP("mlp_improved", "")
improved_mlp.data = dataset.load()
# Important: modify your Method_MLP to use AdamW + LabelSmoothingCrossEntropy BEFORE running this line
improved_metrics = improved_mlp.run()
improved_mlp.plot_learning_curves()


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

    # Create directory if it doesn't exist
    save_dir = '../../result/stage_2_result/'
    os.makedirs(save_dir, exist_ok=True)

    # Save the figure
    plt.savefig(os.path.join(save_dir, 'learning_curves_comparison.png'))
    print("Saved learning_curves_comparison.png")

plot_learning_curves_comparison(original_mlp, "Original", improved_mlp, "Improved")
end = time.time()
print(f"Time elapsed to train second model: {end - mid:.2f} seconds")