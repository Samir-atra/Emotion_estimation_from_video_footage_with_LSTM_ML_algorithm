"""
augmenting the dataset and adding the new images to the training set 
and normalizing the dataset after extracting the mean and variance
"""

import math
import numpy as np
import tensorflow as tf
import mediapipe as mp
from utils.csv_writer import csv_writer
from data.data_cleaning import sets_cleaner
from mediapipe_tools.visualizing_and_setup import detector
from dotenv import load_dotenv
import os
import sys

load_dotenv()

training_set_hus = sets_cleaner(os.getenv("TRAIN_DATASET"))


def augment_load_data(training_set_hus):
    """
    load the training dataset to be augmented

    Args:
        training_set_hus (list): list of the training set

    Returns:
        training_images (list): list of the training images
        training_labels (list): list of the training labels
    """

    training_images = []
    training_labels = []

    # create an image list and a labels list for the training dataset
    for i in range(math.floor(len(training_set_hus))):
        image = (
            np.array(training_set_hus[i][1].split()).reshape(48, 48, 1).astype(np.uint8)
        )
        training_images.append(image)
        training_labels.append(int(training_set_hus[i][0]))

    return training_images, training_labels


def augmentation_models():
    """
    Builds three sequential models for rescaling and augmenting images

    Returns:
      rescaling1: A sequential model for scaling down pixel values
      rescaling2: A sequential model for scaling up pixel values
      augment: A sequential model for image augmentation
    """
    rescaling1 = tf.keras.Sequential([tf.keras.layers.Rescaling(1.0 / 255)])     # scale the image down

    rescaling2 = tf.keras.Sequential([tf.keras.layers.Rescaling(1.0 * 255)])     # scale the image up
    # apply horizontal flip and random rotation to the image
    augment = tf.keras.Sequential(
        [tf.keras.layers.RandomFlip("horizontal"), tf.keras.layers.RandomRotation(0.1)]
    )
    return rescaling1, rescaling2, augment


def augment_images(training_images, training_labels, rescaling1, rescaling2, augment):
    """
    augments the images in the training set while taking a non-optimal ay for, 
    processing them and checking their calidity to be detected by mediapipefor more check the blog: 
    https://medium.com/@samiratra95/image-augmentation-using-tensorflow-and-mediapipe-baf54651f9fc

    Args:
        training_images (list): the training set images
        rescaling1 (keras.sequential): the first rescaling down model
        rescaling2 (keras.sequential): the second rescaling up model
        augment (keras.sequential): the augmentation model

    Returns:
        augmented_training_set (list): the augmented training set
    """
    augmented_training_set = []

    for ele in range(len(training_images)):
        img = training_images[ele]
        label = training_labels[ele]
        image = rescaling1(img)
        aug_image = augment(image)         # apply augmentations on the image
        aug_image = rescaling2(aug_image)
        aug_image = tf.cast(aug_image, tf.uint8)        # cast the pixels values into integers
        aug_image = np.array(aug_image)
        flatten_image = (
            aug_image.flatten()
        )  
        # flatten the augmented image, to be used in creating the csv file
        flat_aug_image = [flatten_image[i] for i in range(0, len(flatten_image))]
        frame = mp.Image(image_format=mp.ImageFormat.SRGB, data=aug_image)
        detection_result = detector()
        
        # check if the face in the image are detectable with mediapipe
        detection_result = detection_result.detect(frame) 
        if detection_result.face_blendshapes == []:
            continue
        else:
            element = [training_labels[ele]]
            for i in flat_aug_image:
                element.append(i)
            augmented_training_set.append(
                [
                    element[0],
                    str(element[1:]).replace(",", "").replace("[", "").replace("]", ""),
                    "Training",
                ]
            )
    csv_writer("training_set_full.csv", ['emotion','pixels'], augmented_training_set)

    return augmented_training_set


def normalize(x_train, x_val):
    """
    Normalizes the images of the dataset, and find the mean and standard deviation.
    for the normalization layer the mean and standard diviation were found using the 
    adapt function instead of finding the mean and standard deviation manually.


    Args:
        X_train (list): the training set
        X_val (list): the validation set

    Returns:
        NX_train (list): normalized training images
        NX_val (list): normalized validation set
    """

    mean = tf.math.reduce_mean(x_train, axis=0)            # find the mean of the dataset for the normalization layer
    stddev = tf.math.reduce_std(x_train, axis=0)          # find the standard deviation
    mean = np.array(mean).T
    stddev = np.array(stddev).T

    # normalize data
    norm = tf.keras.layers.Normalization(axis=1)
    norm.adapt(x_train)
    xx_train = norm(x_train)
    nx_train = np.array(xx_train)

    norm.adapt(x_train)
    x_val = norm(x_val)
    nx_val = np.array(x_val)

    return nx_train, nx_val
