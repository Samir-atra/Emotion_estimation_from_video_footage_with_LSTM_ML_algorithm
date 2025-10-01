"""
opens the blendshapes dataset files and load it to be prepared 
for training the LSTM model and plotting the features by classes
"""


import csv
import numpy as np
import tensorflow as tf
from dotenv import load_dotenv
import os

load_dotenv()

def load_training_data():
    """
    loads the blendshapes of the training set

    Return:
        x_train (np.array): the blendshapes of the training set
        y_train (np.array): the labels of the training set
    """

    traindata = []
    blend_set = []
    labels_set = []

    with open(os.getenv("TRAIN_DATASET"),
        mode="r", encoding="utf-8"
    ) as data:
        csvfile = csv.reader(data)
        next(csvfile)
        for line in csvfile:
            traindata.append(line[:])
        np.random.shuffle(traindata)
        for lines in traindata:
            blend_set.append(lines[0:52])    # load the full set of the 52 mediapipe blendshapes
            labels_set.append(lines[52])
    blends_set = np.array(
        blend_set, dtype=np.float64
    )  # create an array of the blendshapes
    labels_set = np.array(labels_set, dtype=np.float64)  # create an array of the labels

    # Reshaping Array
    x_train = np.reshape(blends_set, (22515, 52, 1))        # reshaping the full blendshapes training dataset 
    y_train = np.reshape(labels_set, (22515, 1)).astype("int")
    y_train = tf.keras.utils.to_categorical(
        y_train, num_classes=3
    )  # encode the labels as one-hot to be fit for the categorical cross entropy loss function

    return x_train, y_train


def load_validation():
    """
    load the validation set for the LSTM model

    Returns:
        x_val (np.array): the blendshapes of the validation set
        y_val (np.array): the labels of the validation set
    """

    valdata = []
    val_blend_set = []
    val_labels_set = []
    with open(os.getenv("VAL_DATASET"),
        mode="r", encoding="utf-8"
    ) as val_data:
        csvfile = csv.reader(val_data)
        next(csvfile)
        for line in csvfile:
            valdata.append(line[:])
        np.random.shuffle(valdata)                 # shuffle the dataset rows
        for lines in valdata:
            val_blend_set.append(lines[0:52])      # load the validation blendshapes set
            val_labels_set.append(lines[52])
    val_blend_set = np.array(val_blend_set, dtype=np.float64)
    val_labels_set = np.array(val_labels_set, dtype=np.float64)
    x_val = np.reshape(val_blend_set, (1657, 52, 1))              
    y_val = np.reshape(val_labels_set, (1657, 1)).astype("int")
    y_val = tf.keras.utils.to_categorical(y_val, num_classes=3)

    return x_val, y_val
