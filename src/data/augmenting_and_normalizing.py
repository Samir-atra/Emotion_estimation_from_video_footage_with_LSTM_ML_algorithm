"""Module for augmenting and normalizing emotion estimation datasets.

This module provides functionalities for:
- Loading and preparing training data for augmentation.
- Building Keras models for image rescaling and augmentation.
- Applying augmentation techniques to images and filtering them based on MediaPipe face detection.
- Normalizing dataset features using TensorFlow's Normalization layer.
"""
import os
import math
import numpy as np
import tensorflow as tf
import keras
import mediapipe as mp
from dotenv import load_dotenv
from utils.csv_writer import csv_writer
from data.data_cleaning import sets_cleaner
from mediapipe_tools.visualizing_and_setup import detector

load_dotenv()

training_set_hus = sets_cleaner(os.getenv("TRAIN_DATASET"))


def augment_load_data(training_set):
    """
    Loads and prepares the training dataset for augmentation.

    This function iterates through the provided `training_set`, extracts image pixel data
    and corresponding labels, and converts the image data into a suitable format for
    further processing (e.g., augmentation).

    Args:
        training_set (list): A list of training instances, where each instance is expected
                             to be a list or array-like object. The first element `[0]` is
                             assumed to be the label, and the second element `[1]` is
                             assumed to be a space-separated string of pixel values.

    Returns:
        tuple: A tuple containing two lists:
            - training_images (list): A list of 48x48 grayscale NumPy arrays (tf.uint8) representing the training images.
            - training_labels (list): A list of integers representing the labels for the training images.
    """

    training_images = []
    training_labels = []

    # Iterate through the training set to extract images and labels
    for i in range(math.floor(len(training_set))):
        # Convert pixel string to NumPy array and reshape to 48x48 grayscale
        image = (
            np.array(training_set[i][1].split()).reshape(48, 48, 1).astype(np.uint8)
        )
        training_images.append(image)
        training_labels.append(int(training_set[i][0]))

    return training_images, training_labels


def augmentation_models():
    """
    Builds and returns three sequential Keras models for image rescaling and augmentation.

    These models are designed to be used in an image augmentation pipeline:
    - `rescaling1`: Scales down pixel values from [0, 255] to [0, 1].
    - `rescaling2`: Scales up pixel values from [0, 1] back to [0, 255].
    - `augment`: Applies random horizontal flipping and random rotation to images.

    Returns:
        tuple: A tuple containing the three Keras Sequential models:
            - rescaling1 (keras.Sequential): Model for scaling down pixel values.
            - rescaling2 (keras.Sequential): Model for scaling up pixel values.
            - augment (keras.Sequential): Model for applying image augmentation transformations.
    """
    # Model to scale image pixel values down to [0, 1]
    rescaling1 = keras.Sequential([keras.layers.Rescaling(1.0 / 255)])

    # Model to scale image pixel values up to [0, 255]
    rescaling2 = keras.Sequential([keras.layers.Rescaling(1.0 * 255)])
    
    # Model to apply random horizontal flip and random rotation for augmentation
    augment = keras.Sequential(
        [keras.layers.RandomFlip("horizontal"), keras.layers.RandomRotation(0.1)]
    )
    return rescaling1, rescaling2, augment


def augment_images(training_images, training_labels, rescaling1, rescaling2, augment):
    """
    Augments images in the training set, processes them, and checks their validity
    using MediaPipe face detection before adding them to the augmented dataset.

    This function applies a series of transformations to each image:
    1. Rescales the image down.
    2. Applies random augmentation (flip, rotation).
    3. Rescales the image back up.
    4. Converts the image to a MediaPipe-compatible format.
    5. Uses MediaPipe's `detector` to check if a face is detectable in the augmented image.
    6. If a face is detected, the augmented image's pixel data and its label are formatted
       and added to the `augmented_training_set`.
    7. Finally, the `augmented_training_set` is written to a CSV file named "training_set_full.csv".

    For more details on the approach, refer to the blog post:
    https://medium.com/@samiratra95/image-augmentation-using-tensorflow-and-mediapipe-baf54651f9fc

    Args:
        training_images (list): A list of original training images (NumPy arrays).
        training_labels (list): A list of labels corresponding to the `training_images`.
        rescaling1 (keras.Sequential): The Keras model for scaling down pixel values.
        rescaling2 (keras.Sequential): The Keras model for scaling up pixel values.
        augment (keras.Sequential): The Keras model for applying image augmentation.

    Returns:
        list: The `augmented_training_set` containing valid augmented image data and labels.
    """
    augmented_training_set = []

    for idx, img in enumerate(training_images):
        image = img
        label = training_labels[idx]
        
        # Apply rescaling and augmentation
        image = rescaling1(image)
        aug_image = augment(image)                   # Apply augmentations on the image
        aug_image = rescaling2(aug_image)
        aug_image = tf.cast(aug_image, tf.uint8)        # Cast pixel values into integers
        aug_image = np.array(aug_image)
        
        # Flatten the augmented image for CSV storage
        flatten_image = aug_image.flatten()
        flat_aug_image = [flatten_image[i] for i in range(0, len(flatten_image))]
        
        # Prepare image for MediaPipe detection
        frame = mp.Image(image_format=mp.ImageFormat.SRGB, data=aug_image)
        face_landmarker_detector = detector()

        # Check if the face in the augmented image is detectable with MediaPipe
        detection_result = face_landmarker_detector.detect(frame)
        if detection_result.face_blendshapes == []:
            continue  # Skip if no face is detected
        else:
            # Format the augmented image data and label for the new training set
            element = [label]
            for i in flat_aug_image:
                element.append(i)
            augmented_training_set.append(
                [
                    element[0],
                    str(element[1:]).replace(",", "").replace("[", "").replace("]", ""),
                    "Training",
                ]
            )
    # Write the augmented training set to a CSV file
    csv_writer("training_set_full.csv", ['emotion','pixels'], augmented_training_set)

    return augmented_training_set


def normalize(x_train, x_val):
    """
    Normalizes the images of the dataset (training and validation sets).

    This function uses Keras's `Normalization` layer to adapt to the training data
    and then applies this normalization to both the training and validation sets.
    The mean and standard deviation for normalization are implicitly calculated
    by the `adapt` function of the `Normalization` layer.

    Args:
        x_train (tf.Tensor or np.array): The training set features.
        x_val (tf.Tensor or np.array): The validation set features.

    Returns:
        tuple: A tuple containing the normalized training and validation sets:
            - nx_train (np.array): Normalized training images.
            - nx_val (np.array): Normalized validation images.
    """

    # Note: The mean and stddev are calculated by norm.adapt(x_train) internally.
    # Manual calculation of mean and stddev is not strictly necessary if using adapt.
    # mean = tf.math.reduce_mean(x_train, axis=0)
    # stddev = tf.math.reduce_std(x_train, axis=0)
    # mean = np.array(mean).T
    # stddev = np.array(stddev).T

    # Initialize and adapt the Normalization layer to the training data
    norm = keras.layers.Normalization(axis=1)
    norm.adapt(x_train)
    
    # Apply normalization to training and validation sets
    xx_train = norm(x_train)
    nx_train = np.array(xx_train)

    # Re-adapting for x_val is redundant if x_train is representative of the data distribution.
    # Typically, the same normalization parameters (learned from x_train) are applied to x_val.
    # norm.adapt(x_train) # This line is kept as per original logic, but consider if it's intended.
    x_val = norm(x_val)
    nx_val = np.array(x_val)

    return nx_train, nx_val
