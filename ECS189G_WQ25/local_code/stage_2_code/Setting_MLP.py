import sys
import os

# Add the correct parent directory of 'local_code' to sys.path
# modify this if needed
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../ECS189G_WQ25")))

from local_code.stage_2_code.Dataset_Loader import Dataset_Loader
from local_code.stage_2_code.Method_MLP import Method_MLP

# edit this path for different people
dataset = Dataset_Loader("digit_recognition", "", "/home/xhangc/ss1_ECS/ECS189G/ECS189G_WQ25/data/stage_2_data/train.csv", "/home/xhangc/ss1_ECS/ECS189G/ECS189G_WQ25/data/stage_2_data/test.csv")
data = dataset.load()

mlp = Method_MLP("mlp_mnist", "")
mlp.data = data
mlp.run()
mlp.plot_learning_curves()