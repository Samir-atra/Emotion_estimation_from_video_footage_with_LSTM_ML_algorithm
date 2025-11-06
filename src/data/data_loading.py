"""Module for loading and preparing blendshapes dataset for LSTM model training.

This module handles:
- Opening and loading blendshape data from CSV files.
- Preparing training and validation datasets, including shuffling and one-hot encoding labels.
- Providing data in a format suitable for LSTM model input.
"""


import csv
import numpy as np
import tensorflow as tf
from dotenv import load_dotenv
import os

load_dotenv()

def load_training_data():
    """
    Loads the blendshapes and corresponding labels for the training set.

    This function reads data from the CSV file specified by the `TRAIN_DATASET`
    environment variable. It shuffles the data, extracts blendshapes (features)
    and emotion labels, and then reshapes and one-hot encodes the labels
    to be suitable for a categorical cross-entropy loss function in Keras.

    Returns:
        tuple: A tuple containing:
            - x_train (np.array): A NumPy array of shape (num_samples, 52, 1)
                                  representing the blendshapes of the training set.
            - y_train (np.array): A NumPy array of one-hot encoded labels for the training set.
    """

    traindata = []
    blend_set = []
    labels_set = []

    # Open and read the training dataset CSV file
    with open(os.getenv("TRAIN_DATASET"),
        mode="r", encoding="utf-8"
    ) as data:
        csvfile = csv.reader(data)
        next(csvfile)  # Skip the header row
        for line in csvfile:
            traindata.append(line[:])
        
        np.random.shuffle(traindata)  # Shuffle the training data
        
        # Extract blendshapes and labels
        for lines in traindata:
            blend_set.append(lines[0:52])    # Load the full set of 52 MediaPipe blendshapes
            labels_set.append(lines[52])     # Load the corresponding emotion label
    
    # Convert lists to NumPy arrays with float64 dtype
    blends_set = np.array(blend_set, dtype=np.float64)
    labels_set = np.array(labels_set, dtype=np.float64)

    # Reshape arrays for LSTM model input
    x_train = np.reshape(blends_set, (22515, 52, 1))        # Reshaping the blendshapes training dataset 
    y_train = np.reshape(labels_set, (22515, 1)).astype("int")
    
    # One-hot encode the labels for categorical cross-entropy loss
    y_train = tf.keras.utils.to_categorical(y_train, num_classes=3)

    return x_train, y_train


def load_validation():
    """
    Loads the blendshapes and corresponding labels for the validation set.

    This function reads data from the CSV file specified by the `VAL_DATASET`
    environment variable. It shuffles the data, extracts blendshapes (features)
    and emotion labels, and then reshapes and one-hot encodes the labels
    to be suitable for a categorical cross-entropy loss function in Keras.

    Returns:
        tuple: A tuple containing:
            - x_val (np.array): A NumPy array of shape (num_samples, 52, 1)
                                representing the blendshapes of the validation set.
            - y_val (np.array): A NumPy array of one-hot encoded labels for the validation set.
    """

    valdata = []
    val_blend_set = []
    val_labels_set = []
    
    # Open and read the validation dataset CSV file
    with open(os.getenv("VAL_DATASET"),
        mode="r", encoding="utf-8"
    ) as val_data:
        csvfile = csv.reader(val_data)
        next(csvfile)  # Skip the header row
        for line in csvfile:
            valdata.append(line[:])
        
        np.random.shuffle(valdata)                 # Shuffle the dataset rows
        
        # Extract blendshapes and labels
        for lines in valdata:
            val_blend_set.append(lines[0:52])      # Load the validation blendshapes set
            val_labels_set.append(lines[52])       # Load the corresponding emotion label
    
    # Convert lists to NumPy arrays with float64 dtype
    val_blend_set = np.array(val_blend_set, dtype=np.float64)
    val_labels_set = np.array(val_labels_set, dtype=np.float64)
    
    # Reshape arrays for LSTM model input
    x_val = np.reshape(val_blend_set, (1657, 52, 1))              
    y_val = np.reshape(val_labels_set, (1657, 1)).astype("int")
    
    # One-hot encode the labels for categorical cross-entropy loss
    y_val = tf.keras.utils.to_categorical(y_val, num_classes=3)

    return x_val, y_val
