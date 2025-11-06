"""Module for processing and balancing emotion recognition datasets.

This module provides functionalities to:
- Count images per class and identify images unreadable by MediaPipe.
- Create a class-balanced dataset with a reduced number of emotion categories (Happy, Sad, Unknown).
"""

import csv
import mediapipe as mp
import matplotlib.pyplot as plt
import tensorflow as tf
from mediapipe_tools.visualizing_and_setup import detector
from dotenv import load_dotenv
import os

load_dotenv()

def categories_and_unreadable_counter(fer2013_path):
    """
    Counts the number of images in each emotion class and the number of images
    that MediaPipe cannot detect a face in.

    This function reads an image dataset (e.g., FER2013), iterates through each image,
    and performs the following:
    1. Increments a counter for each emotion class.
    2. Processes the image to be compatible with MediaPipe.
    3. Uses MediaPipe's face detector to check for blendshapes.
    4. If no face is detected, increments a 'skipped' counter for that emotion class.

    Args:
        fer2013_path (str): The absolute path to the FER2013 dataset CSV file.

    Prints:
        - The counts of images per category.
        - The counts of skipped (unreadable by MediaPipe) images per category.
    """
    # Initialize dictionaries to count images per category and skipped images
    categories_counts = {"0": 0, "1": 0, "2": 0, "3": 0, "4": 0, "5": 0, "6": 0}
    skipped = {"0": 0, "1": 0, "2": 0, "3": 0, "4": 0, "5": 0, "6": 0}

    # Read the dataset file using TensorFlow's file I/O
    data = tf.io.read_file(fer2013_path)
    # Split the dataset into individual lines (rows)
    f = tf.strings.split(data, sep="\n")
    
    # Iterate through each line (instance) in the dataset, skipping the header and last empty line
    for lines in f[1:-1]:
        # Extract the emotion label (key) from the current line
        key = str(tf.strings.as_string(lines).numpy().decode("utf-8")).split(",")[0]
        if key in categories_counts.keys():
            categories_counts[key] = categories_counts[key] + 1
        
        # Process the string of pixels into an image format suitable for MediaPipe
        image_pixels_str = (
            str(tf.strings.as_string(lines).numpy().decode("utf-8"))
            .split(",")[1]
            .split(" ")
        )
        
        # Convert pixel strings to TensorFlow tensor, reshape, and convert to RGB
        image_tensor = tf.convert_to_tensor(image_pixels_str)
        image_tensor = tf.make_tensor_proto(image_tensor, dtype=tf.uint8)
        image_array = tf.make_ndarray(image_tensor).reshape(48, 48, 1)
        image_tensor_uint8 = tf.convert_to_tensor(image_array, dtype=tf.uint8)
        rgb_frame_data = tf.image.grayscale_to_rgb(image_tensor_uint8).numpy()
        
        # Create MediaPipe Image object
        rgb_frame = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame_data)
        
        # Detect the face in the image using MediaPipe detector
        face_landmarker_detector = detector()
        detection_result = face_landmarker_detector.detect(rgb_frame)
        
        # Check if MediaPipe was able to detect a face (i.e., blendshapes are present)
        if detection_result.face_blendshapes == []:
            skipped[key] = skipped[key] + 1  # Increment skipped counter if no face detected
        else:
            continue  # Continue to the next image if a face is detected
            
    print("Image counts per category:", categories_counts)
    print("Skipped image counts per category (unreadable by MediaPipe):", skipped)


def balanced_dataset(full_set_path):
    """
    Creates a class-balanced dataset by sampling a limited number of images
    from each of the original seven emotion classes and remapping them into
    a three-class system (Happy, Sad, Unknown).

    This function reads a full dataset, iterates through its instances, and selectively
    appends images to a new dataset (`fullset`) based on class limits to achieve balance.
    Original emotion labels are remapped as follows:
    - Original classes 0, 1, 2, 5, 6 are remapped to '1' (e.g., Sad, Fear, Disgust, Surprise, Neutral).
    - Original class 3 (Happy) is remapped to '0'.
    - Original class 4 (Angry) is remapped to '2'.

    Args:
        full_set_path (str): The absolute path to the full dataset CSV file.

    Returns:
        list: A list representing the new class-balanced dataset with remapped labels.
    """

    fullset = []
    class_counter = {"0": 0, "1": 0, "2": 0, "3": 0, "4": 0, "5": 0, "6": 0}

    with open(full_set_path, mode="r", encoding="utf-8") as data:
        csvfile = csv.reader(data)
        next(csvfile)  # Skip the header row
        
        # Iterate over the dataset and append images to a list, applying class limits and remapping labels
        for lines in csvfile:
            # Remap and limit images for class '1' (Sad, Fear, Disgust, Surprise, Neutral)
            if lines[0] == "0" and class_counter["0"] < 1500:
                class_counter[lines[0]] = class_counter[lines[0]] + 1
                lines[0] = "1"  # Remap to '1'
                fullset.append(lines)
            elif lines[0] == "1" and class_counter["1"] < 1500:
                class_counter[lines[0]] = class_counter[lines[0]] + 1
                lines[0] = "1"  # Remap to '1'
                fullset.append(lines)
            elif lines[0] == "2" and class_counter["2"] < 1500:
                class_counter[lines[0]] = class_counter[lines[0]] + 1
                lines[0] = "1"  # Remap to '1'
                fullset.append(lines)
            # Remap and limit images for class '0' (Happy)
            elif lines[0] == "3" and class_counter["3"] < 4000:
                class_counter[lines[0]] = class_counter[lines[0]] + 1
                lines[0] = "0"  # Remap to '0'
                fullset.append(lines)
            # Remap and limit images for class '2' (Angry)
            elif lines[0] == "4" and class_counter["4"] < 4000:
                class_counter[lines[0]] = class_counter[lines[0]] + 1
                lines[0] = "2"  # Remap to '2'
                fullset.append(lines)
            # Remap and limit images for class '1' (Sad, Fear, Disgust, Surprise, Neutral)
            elif lines[0] == "5" and class_counter["5"] < 1500:
                class_counter[lines[0]] = class_counter[lines[0]] + 1
                lines[0] = "1"  # Remap to '1'
                fullset.append(lines)
            elif lines[0] == "6" and class_counter["6"] < 1500:
                class_counter[lines[0]] = class_counter[lines[0]] + 1
                lines[0] = "1"  # Remap to '1'
                fullset.append(lines)

    print("Class distribution after balancing and remapping:", class_counter)
    return fullset


# Example usage when run as a script
# if __name__ == "__main__":
#     categories_and_unreadable_counter(os.getenv("FER2013"))
#     balanced_dataset(os.getenv("TRAIN_DATASET"))
