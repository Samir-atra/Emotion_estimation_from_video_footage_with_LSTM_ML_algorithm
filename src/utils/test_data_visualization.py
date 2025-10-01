"""
Visualize the blendshapes from the test dataset to understand their distribution
across different emotion classes.
"""

import csv
import matplotlib.pyplot as plt
import numpy as np
from dotenv import load_dotenv
import os
import sys

load_dotenv()

def visualize_blendshapes(dataset_path):
    """
    Loads blendshapes data from a CSV file, prints the dimensions,
    and visualizes the distribution of each blendshape per emotion class.

    Args:
        dataset_path (str): The path to the blendshapes CSV dataset.
    """
    all_data = []
    with open(dataset_path, mode="r", encoding="utf-8") as data_file:
        csv_reader = csv.reader(data_file)
        next(csv_reader)  # Skip header row
        for row in csv_reader:
            all_data.append(row)

    # Convert to numpy array for easier slicing
    data_array = np.array(all_data, dtype=np.float64)

    # Print the number of rows and columns
    rows, cols = data_array.shape
    print(f"Dataset loaded from: {dataset_path}")
    print(f"Number of rows: {rows}")
    print(f"Number of columns: {cols}")

    # Separate data by class (emotion label is in column 52, index 53)
    # Columns: 0-51 are blendshapes, 52 is emotion, 53 is index
    class_0 = data_array[data_array[:, 52] == 0]
    class_1 = data_array[data_array[:, 52] == 1]
    class_2 = data_array[data_array[:, 52] == 2]

    print(f"\nData points per class:")
    print(f"Class 0: {len(class_0)}")
    print(f"Class 1: {len(class_1)}")
    print(f"Class 2: {len(class_2)}")

    # Create a grid of plots for each blendshape
    plt.figure(figsize=[25, 25])
    plt.suptitle("Blendshape Scores Distribution by Emotion Class", fontsize=20)

    num_blendshapes = 52
    for i in range(num_blendshapes):
        plt.subplot(9, 6, i + 1)
        
        # Plotting distributions using boxplots for clarity
        data_to_plot = []
        if len(class_0) > 0:
            data_to_plot.append(class_0[:, i])
        if len(class_1) > 0:
            data_to_plot.append(class_1[:, i])
        if len(class_2) > 0:
            data_to_plot.append(class_2[:, i])

        plt.boxplot(data_to_plot, labels=[f'C{j}' for j in range(len(data_to_plot))])
        plt.title(f"Blendshape {i}")
        plt.ylabel("Score")
        plt.xticks(fontsize=8)
        plt.yticks(fontsize=8)

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.show()


if __name__ == "__main__":
    # The path to your test dataset
    TEST_DATASET_PATH = os.getenv("TEST_DATASET")
    visualize_blendshapes(TEST_DATASET_PATH)