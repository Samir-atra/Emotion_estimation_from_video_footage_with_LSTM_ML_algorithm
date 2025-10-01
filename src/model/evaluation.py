"""load the test set and evaluate the model on it"""

import csv
import numpy as np
import keras
from dotenv import load_dotenv
import os
import sys

load_dotenv()



model = keras.models.load_model(
    "/home/samer/Desktop/Research/Emotion estimation using LSTM/epoch4437val_loss0.6506.keras"
)  # load the model from the .keras file

# plot the model structure in a .png image
keras.utils.plot_model(
    model,
    show_dtype=True,
    show_layer_names=True,
    expand_nested=True,
    show_layer_activations=True,
    show_trainable=True,
    show_shapes=True,
    rankdir="LR",
    to_file="foto.png",
)



test_blend_set = []
test_labels_set = []
test_index_set = []
with open(os.getenv("TEST_DATASET"), mode="r", encoding="utf-8") as test_data:
    csvFile = csv.reader(test_data)
    next(csvFile)
    for lines in csvFile:
        test_blend_set.append(lines[0:52])       # create a list of the images in the test set
        test_labels_set.append(lines[52])        # a list of the labels of the test set
        test_index_set.append(lines[53])         # a list of the indecies of the test set
test_blend_set = np.array(test_blend_set, dtype=np.float64)
test_labels_set = np.array(test_labels_set, dtype=np.float64).astype("int")
test_labels_set = keras.utils.to_categorical(test_labels_set, num_classes=3)     # create one-hot vectors for the labels of the dataset
X_test = np.reshape(test_blend_set, (1646, 52, 1))

model.evaluate(X_test, test_labels_set)   # evaluate the model performance using the test set


# model.save("LSTM_model_full_data_acc:63_f1:55.keras")    # save the tested model as a .keras file
