"""
opens the blendshapes dataset files and load it to be prepared 
for training the LSTM model and plotting the features by classes
"""

# the visualization part is to be separated later

import numpy as np
import csv
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf

  
# load data for LSTM model

traindata = []
blend_set = []
labels_set = []
class1=[]
class2=[]
class3=[]
with open("blends_train_full_set.csv", mode= "r") as data:
  csvFile = csv.reader(data)
  next(csvFile)
  for line in csvFile:
    traindata.append(line[:])
  np.random.shuffle(traindata)
  for lines in traindata:
      blend_set.append(lines[0:52])
      labels_set.append(lines[52])
blends_set = np.array(blend_set, dtype=np.float64)        # create an array of the blendshapes
labels_set = np.array(labels_set, dtype=np.float64)       # create an array of the labels

#Reshaping Array
X_train = np.reshape(blends_set, (22515, 52,1))
Y_train = np.reshape(labels_set, (22515,1)).astype('int')
Y_train = tf.keras.utils.to_categorical(Y_train, num_classes=3)       # encode the labels as one-hot


# load validation data
valdata = []
val_blend_set = []
val_labels_set = []
with open("blends_val_all_emotion.csv", mode= "r") as val_data:
  csvFile = csv.reader(val_data)
  next(csvFile)
  for line in csvFile:
    valdata.append(line[:])
  np.random.shuffle(valdata)
  for lines in valdata:
      val_blend_set.append(lines[0:34])
      val_labels_set.append(lines[34])
val_blend_set = np.array(val_blend_set, dtype=np.float64)
val_labels_set = np.array(val_labels_set, dtype=np.float64)
X_val = np.reshape(val_blend_set, (1657, 34,1))
y_val = np.reshape(val_labels_set, (1657,1)).astype('int')
y_val = tf.keras.utils.to_categorical(y_val, num_classes=3)


  
  
  