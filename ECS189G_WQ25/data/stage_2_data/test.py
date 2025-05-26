import pandas as pd

# Load the data
train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")

# Show basic structure
print("Train shape:", train_df.shape)
print("Test shape:", test_df.shape)

# Show column names and first few rows
print("\nTrain columns:", train_df.columns.tolist())
print("\nFirst few rows of train data:")
print(train_df.head())

# Check if last column is the label (common format)
print("\nLabel value counts (train):")
print(train_df.iloc[:, -1].value_counts())
