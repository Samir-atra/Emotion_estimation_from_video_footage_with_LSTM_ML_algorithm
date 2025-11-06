"""Module for evaluating the trained LSTM model on the test set.

This script loads a pre-trained Keras model, visualizes its architecture,
loads and preprocesses the test dataset, and then evaluates the model's
performance using the test data.
"""

import csv
import numpy as np
import keras
from dotenv import load_dotenv
import os
import sys

load_dotenv()


# Load the pre-trained Keras model from the specified .keras file
model = keras.models.load_model(
    "/home/samer/Desktop/Research/Emotion estimation using LSTM/epoch4437val_loss0.6506.keras"
)

# Plot the model structure and save it as a .png image for visualization
keras.utils.plot_model(
    model,
    show_dtype=True,
    show_layer_names=True,
    expand_nested=True,
    show_layer_activations=True,
    show_trainable=True,
    show_shapes=True,
    rankdir="LR",
    to_file="foto.png", # Output file name for the model plot
)


# Initialize lists to store test dataset components
test_blend_set = []  # Stores blendshape features
test_labels_set = [] # Stores emotion labels
test_index_set = []  # Stores original indices of the test samples

# Load the test dataset from the CSV file specified in environment variables
with open(os.getenv("TEST_DATASET"), mode="r", encoding="utf-8") as test_data:
    csvFile = csv.reader(test_data)
    next(csvFile)  # Skip the header row
    for lines in csvFile:
        test_blend_set.append(lines[0:52])       # Extract blendshape features (first 52 columns)
        test_labels_set.append(lines[52])        # Extract emotion label (53rd column)
        test_index_set.append(lines[53])         # Extract original index (54th column)

# Convert lists to NumPy arrays with appropriate data types
test_blend_set = np.array(test_blend_set, dtype=np.float64)
test_labels_set = np.array(test_labels_set, dtype=np.float64).astype("int")

# One-hot encode the labels to be compatible with categorical cross-entropy loss
test_labels_set = keras.utils.to_categorical(test_labels_set, num_classes=3)

# Reshape the blendshape data for LSTM model input (samples, timesteps, features)
X_test = np.reshape(test_blend_set, (1646, 52, 1))

# Evaluate the model's performance on the preprocessed test set
model.evaluate(X_test, test_labels_set)


# model.save("LSTM_model_full_data_acc:63_f1:55.keras")    # Example: Save the tested model as a .keras file (commented out)
