from local_code.stage_5_code.GCN_modifed import Method_GCN
from local_code.stage_5_code.Dataset_Loader_Node_Classification import Dataset_Loader
import time
import os
from tabulate import tabulate
import numpy as np
import matplotlib.pyplot as plt

# Configuration
DATASETS = ['citeseer', 'cora', 'pubmed']  # Available datasets
DATASETS_TO_RUN = ['citeseer', 'cora', 'pubmed']  # To run all


def print_results_table(results):
    """
    Print results in a formatted table

    Args:
        results (dict): Dictionary containing results for each dataset
    """
    table_data = []
    headers = ["Dataset", "Accuracy", "Precision", "Recall", "F1-Score", "Time (s)"]

    for dataset_name, result in results.items():
        if 'error' in result:
            row = [dataset_name, "Error", "Error", "Error", "Error", "N/A"]
        else:
            metrics = result['metrics']
            row = [
                dataset_name,
                f"{metrics['accuracy']:.4f}",
                f"{metrics.get('precision_macro', 0.0):.4f}",
                f"{metrics.get('recall_macro', 0.0):.4f}",
                f"{metrics.get('f1_macro', 0.0):.4f}",
                f"{result['execution_time']:.2f}"
            ]
        table_data.append(row)

    print("\nResults Summary:")
    print(tabulate(table_data, headers=headers, tablefmt="grid"))


def run_gcn(dataset_name):
    """
    Run GCN on a specific dataset

    Args:
        dataset_name (str): Name of the dataset ('citeseer', 'cora', or 'pubmed')

    Returns:
        dict: Dictionary containing the evaluation metrics and learning curves data
        float: Time taken for execution
    """
    start = time.time()

    # Initialize dataset
    dataset = Dataset_Loader(dName=dataset_name)

    # Set up dataset path
    current_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_path = os.path.join(current_dir, '..', '..', 'data', 'stage_5_data', dataset_name)

    # Ensure the directory exists
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset directory not found: {dataset_path}\n"
                                f"Please ensure the {dataset_name} dataset is present in this location")

    dataset.dataset_source_folder_path = dataset_path

    # Load data
    data = dataset.load()

    # Initialize and run model
    model = Method_GCN('GCN', f'Graph Convolutional Network for Node Classification on {dataset_name}')
    model.data = data

    print(f'\n=== Running GCN on {dataset_name} dataset ===')
    metrics = model.run()

    # Plot learning curves with dataset name
    print(f'--plotting learning curves for {dataset_name}...')

    model.plot_learning_curves(f"{dataset_name} modified")

    execution_time = time.time() - start
    print(f"Time elapsed: {execution_time:.2f} seconds\n")

    # Store learning curves data
    learning_curves = {
        'train_loss': model.loss_history,
        'val_loss': model.val_loss_history,
        'train_acc': model.accuracy_history,
        'val_acc': model.val_accuracy_history
    }

    return metrics, execution_time, learning_curves


def plot_combined_curves(results):
    """
    Plot combined learning curves for all datasets
    """
    if len(results) > 1:  # Only create combined plots if there's more than one dataset
        # Create subplots for loss and accuracy
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Plot loss curves
        for dataset_name, result in results.items():
            if 'learning_curves' in result:
                curves = result['learning_curves']
                ax1.plot(curves['train_loss'], label=f'{dataset_name} (Train)')
                ax1.plot(curves['val_loss'], label=f'{dataset_name} (Val)', linestyle='--')

        ax1.set_title('Combined Loss Curves')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)

        # Plot accuracy curves
        for dataset_name, result in results.items():
            if 'learning_curves' in result:
                curves = result['learning_curves']
                ax2.plot(curves['train_acc'], label=f'{dataset_name} (Train)')
                ax2.plot(curves['val_acc'], label=f'{dataset_name} (Val)', linestyle='--')

        ax2.set_title('Combined Accuracy Curves')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
        ax2.grid(True)

        plt.tight_layout()
        plt.savefig('gcn_learning_curves_combined.png', dpi=300, bbox_inches='tight')
        plt.close()


def main():
    """
    Main function to run GCN on specified datasets
    """
    # Validate datasets to run
    invalid_datasets = [d for d in DATASETS_TO_RUN if d not in DATASETS]
    if invalid_datasets:
        raise ValueError(f"Invalid dataset(s): {invalid_datasets}. "
                         f"Available datasets are: {DATASETS}")

    # Store results for all runs
    results = {}

    print(f"Starting GCN experiments on {len(DATASETS_TO_RUN)} dataset(s)...")
    print("=" * 50)

    # Run GCN on each specified dataset
    for dataset_name in DATASETS_TO_RUN:
        try:
            metrics, execution_time, learning_curves = run_gcn(dataset_name)
            results[dataset_name] = {
                'metrics': metrics,
                'execution_time': execution_time,
                'learning_curves': learning_curves
            }
        except Exception as e:
            print(f"\nError running GCN on {dataset_name}:")
            print(f"Error message: {str(e)}")
            results[dataset_name] = {'error': str(e)}

    # Plot combined learning curves
    if len(DATASETS_TO_RUN) > 1:
        print("\nGenerating combined learning curves...")
        plot_combined_curves(results)
        print("Saved combined learning curves as 'gcn_learning_curves_combined.png'")

    # Print final results in table format
    print_results_table(results)

    return results


if __name__ == '__main__':
    main()