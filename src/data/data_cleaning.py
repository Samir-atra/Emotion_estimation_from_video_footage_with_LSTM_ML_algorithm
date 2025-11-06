"""Module for cleaning image datasets by filtering out images not detectable by MediaPipe.

This module provides functions to:
- Split a full dataset into training, validation, and test sets.
- Clean datasets by removing images where MediaPipe's face detector cannot find blendshapes.
- Write the cleaned dataset splits into new CSV files.
"""

import os
import tensorflow as tf
import mediapipe as mp
from dotenv import load_dotenv
from data_processing import balanced_dataset
from mediapipe_tools.visualizing_and_setup import detector
from utils.csv_writer import csv_writer

load_dotenv()


def list_creator(full_data_set):
    """
    Splits a comprehensive dataset into training, validation, and test sets
    based on a 'Usage' field within each data instance.

    Args:
        full_data_set (list): A list of data instances, where each instance is expected
                              to be a list or array-like object. The element at index `[2]`
                              is assumed to contain the usage type (e.g., "Training",
                              "PublicTest", "PrivateTest").

    Returns:
        tuple: A tuple containing three lists:
            - training_set_list (list): Instances marked as "Training".
            - validation_set_list (list): Instances marked as "PublicTest".
            - test_set_list (list): Instances marked as "PrivateTest".
    """
    # Initialize empty lists for the dataset splits
    training_set_list = []
    validation_set_list = []
    test_set_list = []
    
    # Iterate through the full dataset and categorize instances into respective lists
    for i in full_data_set:
        if i[2] == "Training":                # Assign to training split
            training_set_list.append(i)
        elif i[2] == "PublicTest":            # Assign to validation split
            validation_set_list.append(i)
        elif i[2] == "PrivateTest":           # Assign to test split
            test_set_list.append(i)

    return training_set_list, validation_set_list, test_set_list


def sets_cleaner(data_set_path):
    """
    Cleans a dataset by filtering out images that are not recognizable by MediaPipe's
    face detector (i.e., no blendshapes are detected).

    This function reads image data from a specified dataset path, processes each image
    to be compatible with MediaPipe, and then uses the MediaPipe detector to check for
    face blendshapes. Only images where blendshapes are detected are included in the
    returned cleaned dataset.

    Args:
        data_set_path (str): The environment variable key (e.g., "TRAIN_DATASET")
                             that points to the path of the dataset CSV file.

    Returns:
        list: A list of cleaned data instances, where each instance's image has been
              successfully processed and detected by MediaPipe.
    """
    data_set = []

    # Read the dataset file using TensorFlow's file I/O
    data = tf.io.read_file(os.getenv(data_set_path))
    f = tf.strings.split(data, sep="\n")
    
    # Loop through each line (instance) in the dataset, skipping the header and last empty line
    for lines in f[1:-1]:
        # Extract pixel data from the CSV line and convert to a list of strings
        image_pixels_str = (
            str(tf.strings.as_string(lines).numpy().decode("utf-8"))
            .split(",")[1]
            .split(" ")
        )
        
        # Convert pixel strings to TensorFlow tensor, reshape, and convert to RGB
        # These steps are necessary for MediaPipe compatibility and are optimized with TensorFlow.
        # For more details: https://medium.com/@samiratra95/image-augmentation-using-tensorflow-and-mediapipe-baf54651f9fc
        image_tensor = tf.convert_to_tensor(image_pixels_str)
        image_tensor = tf.make_tensor_proto(image_tensor, dtype=tf.uint8)
        image_array = tf.make_ndarray(image_tensor).reshape(48, 48, 1)
        image_tensor_uint8 = tf.convert_to_tensor(image_array, dtype=tf.uint8)
        rgb_image = tf.image.grayscale_to_rgb(image_tensor_uint8).numpy()
        
        # Create MediaPipe Image object
        frame = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
        
        # Detect face blendshapes using MediaPipe detector
        face_landmarker_detector = detector()
        detection_result = face_landmarker_detector.detect(frame)
        
        # If blendshapes are detected, add the original data instance to the cleaned set
        if detection_result.face_blendshapes == []:
            continue  # Skip images where no face is detected
        else:
            data_set.append(str(tf.strings.as_string(lines).numpy().decode("utf-8")).split(","))

    return data_set


def write_files(training_data, validation_data, test_data):
    """
    Writes the cleaned training, validation, and test datasets to separate CSV files.

    Args:
        training_data (list): The cleaned training set instances.
        validation_data (list): The cleaned validation set instances.
        test_data (list): The cleaned test set instances.
    """
    fields = ["emotion", "pixels", "Usage"]
    csv_writer("training_set_full.csv", fields, training_data)
    csv_writer("validation_set_full.csv", fields, validation_data)
    csv_writer("test_set_full.csv", fields, test_data)


if __name__=="__main__":
    # Example usage when run as a script
    # Note: The path for balanced_dataset should be an absolute path or correctly resolved.
    fullset = balanced_dataset("/home/samer/Desktop/HAN stuff/Big data Small Data/BDSD/Minor_project/BDSD_Minor_Project/Datasets/training_set_full.csv")
    training_set, validation_set, test_set = list_creator(fullset)
    
    # Clean each dataset split using MediaPipe detection
    training_set_hus = sets_cleaner("TRAIN_DATASET")
    validation_set_hus = sets_cleaner("VAL_DATASET")
    test_set_hus = sets_cleaner("TEST_DATASET")
    
    # Write the cleaned datasets to new CSV files
    write_files(training_set_hus, validation_set_hus, test_set_hus)
