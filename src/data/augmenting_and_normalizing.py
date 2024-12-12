"""
augmenting the dataset and adding the new images to the training set 
and normalizing the dataset after extracting the mean and variance
"""
import math
import numpy as np
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf
from ..utils.csv_writer import csv_writer

# Augmenting the training set images


augmented_training_set = []
training_images = []
training_labels = []

# create an image list and a labels list for the training dataset
for i in range(math.floor(len(training_set_hus))):
    image = np.array(training_set_hus[i][1].split(' ')).reshape(48, 48, 1).astype(np.uint8)
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    training_images.append(image)
    training_labels.append(int(training_set_hus[i][0]))


# rescaling and augmenting images models
rescaling1 = tf.keras.Sequential([ 
  tf.keras.layers.Rescaling(1./255)                         # scale down the images pixel values
])

rescaling2 = tf.keras.Sequential([                          # scale up the pixel values
  tf.keras.layers.Rescaling(1.*255)
])

augment = tf.keras.Sequential([                             # augment by random flipping and random rotation
  tf.keras.layers.RandomFlip("horizontal"),
  tf.keras.layers.RandomRotation(0.1)
])

for ele in range(len(training_images)):
# scale down the image and augment
  img = training_images[ele]
  label = training_labels[ele]
  image = rescaling1(img)
  aug_image = augment(image)
#   scale up the image cast to an integer and transform into a numpy array for it to be understood by mediapipe
  aug_image = rescaling2(aug_image)
  aug_image = tf.cast(aug_image, tf.uint8)
  aug_image = np.array(aug_image)
  flatten_image = aug_image.flatten()                                            # flatten the augmented image, to be used in creating the csv file
  flat_aug_image = [flatten_image[i] for i in range(0,len(flatten_image),3)]
  # flattt = np.reshape(flat_aug_image,(48,48))
  # plt.imshow(flattt)
  # plt.show()
  frame = mp.Image(image_format=mp.ImageFormat.SRGB,data=aug_image)
  detection_result = detector.detect(frame)
  if detection_result.face_blendshapes == []:
    continue
  else:
    element = [training_labels[ele]]
    for i in flat_aug_image:
      element.append(i)
    augmented_training_set.append([element[0],str(element[1:]).replace(',',"").replace('[','').replace(']',''),'Training'])

for images in augmented_training_set:
  image = np.array(images[1].split(' ')).reshape(48, 48, 1).astype(np.uint8)
  image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
  plt.imshow(image)
  plt.show()

csv_writer("training_set_full.csv", ['emotion','pixels'], augmented_training_set)





# data normalization

mean = tf.math.reduce_mean(X_train,axis=0)            # find the mean for the dataset
stddev = tf.math.reduce_std(X_train, axis=0)          # find the standard deviation
mean = np.array(mean).T
stddev = np.array(stddev).T
csv_writer("mean_and_std.csv",'beedoo', mean)
csv_writer("mean_and_std.csv",'beedoo', stddev)

# normalize data
norm = tf.keras.layers.Normalization(axis=1)
norm.adapt(X_train)
print(X_train[0])
XX_train = norm(X_train)
print(XX_train[0])
XX_train = np.array(X_train)
XX_train = X_train[1]
plt.scatter(X_train[1],XX_train)

norm.adapt(X_train)
X_val = norm(X_val)
X_val = np.array(X_val)
