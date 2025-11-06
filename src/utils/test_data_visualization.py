"""Module for visualizing blendshape distributions from the test dataset.

This module provides tools to load blendshape data, analyze its dimensions,
and visualize the distribution of individual blendshape scores across different
emotion classes using box plots. This helps in understanding the characteristics
of the features for each emotion.
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
    Loads blendshapes data from a specified CSV file, prints dataset dimensions,
    and visualizes the distribution of each blendshape score per emotion class.

    The visualization uses box plots to show the spread and central tendency of
    blendshape scores for each of the three emotion classes (0, 1, 2).

    Args:
        dataset_path (str): The absolute path to the blendshapes CSV dataset file.
                            This file is expected to have blendshape scores in columns
                            0-51, emotion labels in column 52, and an index in column 53.
    """
    all_data = []
    # Load the dataset from the specified CSV file
    with open(dataset_path, mode="r", encoding="utf-8") as data_file:
        csv_reader = csv.reader(data_file)
        next(csv_reader)  # Skip the header row
        for row in csv_reader:
            all_data.append(row)

    # Convert the loaded data into a NumPy array for efficient numerical operations
    data_array = np.array(all_data, dtype=np.float64)

    # Print the dimensions of the loaded dataset
    rows, cols = data_array.shape
    print(f"Dataset loaded from: {dataset_path}")
    print(f"Number of rows: {rows}")
    print(f"Number of columns: {cols}")

    # Separate data points based on their emotion class (label in column 52)
    # Assuming 3 classes: 0, 1, 2
    class_0 = data_array[data_array[:, 52] == 0]
    class_1 = data_array[data_array[:, 52] == 1]
    class_2 = data_array[data_array[:, 52] == 2]

    print(f"\nData points per class:")
    print(f"Class 0: {len(class_0)}")
    print(f"Class 1: {len(class_1)}")
    print(f"Class 2: {len(class_2)}")

    # Create a grid of subplots to visualize each blendshape's distribution
    plt.figure(figsize=[25, 25])
    plt.suptitle("Blendshape Scores Distribution by Emotion Class", fontsize=20)

    num_blendshapes = 52 # There are 52 blendshape features (columns 0-51)
    for i in range(num_blendshapes):
        plt.subplot(9, 6, i + 1) # Arrange plots in a 9x6 grid
        
        # Prepare data for box plotting for the current blendshape across classes
        data_to_plot = []
        if len(class_0) > 0:
            data_to_plot.append(class_0[:, i])
        if len(class_1) > 0:
            data_to_plot.append(class_1[:, i])
        if len(class_2) > 0:
            data_to_plot.append(class_2[:, i])

        # Create box plots for the current blendshape, labeled by class
        plt.boxplot(data_to_plot, labels=[f'C{j}' for j in range(len(data_to_plot))])
        plt.title(f"Blendshape {i}") # Title for the current blendshape subplot
        plt.ylabel("Score")
        plt.xticks(fontsize=8)
        plt.yticks(fontsize=8)

    plt.tight_layout(rect=[0, 0, 1, 0.97]) # Adjust layout to prevent titles/labels from overlapping
    plt.show() # Display the generated plots


if __name__ == "__main__":
    # Example usage when the script is executed directly.
    # Loads the test dataset path from environment variables and visualizes blendshapes.
    TEST_DATASET_PATH = os.getenv("TEST_DATASET")
    visualize_blendshapes(TEST_DATASET_PATH)