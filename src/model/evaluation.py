"""load the test set and evaluate the model on it"""

import csv
import numpy as np
import tensorflow as tf


model = tf.keras.models.load_model('/home/samer/Desktop/Big data Small Data/BDSD/Minor_project/BDSD_Minor_Project/trained_models/epoch4437val_loss0.6506.keras')            # load the model to be evaluated
tf.keras.utils.plot_model(model,                      # plot the model
    show_dtype=True,
    show_layer_names=True,
    expand_nested=True,
    show_layer_activations=True,
    show_trainable=True, show_shapes=True, rankdir="LR", to_file='foto.png')

test_blend_set = []
test_labels_set = []
test_index_set = []
with open("blends_test_index.csv", mode= "r") as test_data:             # load the test set
  csvFile = csv.reader(test_data)
  next(csvFile)
  for lines in csvFile:
      test_blend_set.append(lines[0:52])
      test_labels_set.append(lines[52])
      test_index_set.append(lines[53])
test_blend_set = np.array(test_blend_set, dtype=np.float64)
test_labels_set = np.array(test_labels_set, dtype=np.float64).astype('int')
test_labels_set = tf.keras.utils.to_categorical(test_labels_set, num_classes=3)
X_test = np.reshape(test_blend_set, (1646, 52,1))

model.evaluate(X_test,test_labels_set)                          # evaluate the model using the test set


model.save('LSTM_model_full_data_acc:63_f1:55.keras')
