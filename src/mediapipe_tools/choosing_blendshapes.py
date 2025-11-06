"""Module for identifying relevant blendshapes for emotion recognition datasets.

This script focuses on analyzing blendshape occurrences in a given dataset
to determine which blendshapes are most indicative of specific emotions
(e.g., happy, sad, neutral). It uses MediaPipe to detect blendshapes and
applies a threshold to filter for significant activations.
"""

import time
import tensorflow as tf
import mediapipe as mp
from data.data_processing import balanced_dataset
from mediapipe_tools.visualizing_and_setup import detector
from dotenv import load_dotenv
import os
import sys

load_dotenv()

# Configure the GPU to allocate a specific portion of memory for data processing.
# This helps in managing GPU resources and avoiding out-of-memory errors.
gpus = tf.config.list_physical_devices("GPU")
tf.config.experimental.set_synchronous_execution(True) # Enable synchronous execution for easier debugging
if gpus:
    try:
        # Set a logical device configuration with a memory limit for the first GPU
        tf.config.set_logical_device_configuration(
            gpus[0], [tf.config.LogicalDeviceConfiguration(memory_limit=3076)]
        )
        logical_gpus = tf.config.list_logical_devices("GPU")
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Virtual devices must be set before GPUs have been initialized
        print(e)


def choosing_blendshapes(training_set_hus):
    """
    Analyzes the training dataset to count the occurrences of each blendshape
    when its score exceeds a predefined threshold (0.4).

    This function iterates through each image in the `training_set_hus`,
    processes it for MediaPipe detection, and then tallies blendshapes that
    show significant activation. This helps in identifying blendshapes most
    relevant to the emotions present in the dataset.

    Args:
        training_set_hus (list): The training dataset as a Python list, where each
                                 element is expected to contain image pixel data.

    Returns:
        dict: A dictionary where keys are blendshape indices (as strings) and
              values are their respective counts of occurrences above the threshold.
    """

    blend_shapes = {} # Dictionary to store blendshape counts
    # Initialize counts for all 52 blendshapes to zero
    for i in range(0, 52):
        blend_shapes[str(i)] = 0

    counter = 0 # Counter for processed images

    # Iterate through the training set to find relevant blendshapes
    # A threshold of 0.4 is used to determine significance.
    for i in range(len(training_set_hus)):
        # Extract and process image pixels from the dataset instance
        image_pixels_str = str(training_set_hus[i][1]).split(" ")
        image_tensor = tf.convert_to_tensor(image_pixels_str)
        # Serialize the tensor to be processed into a numpy array and reshaped
        image_tensor_proto = tf.make_tensor_proto(image_tensor, dtype=tf.uint8)
        image_array = tf.make_ndarray(image_tensor_proto).reshape(48, 48, 1)
        image_tensor_uint8 = tf.convert_to_tensor(image_array, dtype=tf.uint8)
        rgb_image = tf.image.grayscale_to_rgb(image_tensor_uint8).numpy()
        
        # Create MediaPipe Image object
        frame = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
        
        # Detect face blendshapes using MediaPipe detector
        face_landmarker_detector = detector()
        detection_result = face_landmarker_detector.detect(frame)
        
        # If no face is detected, skip to the next image
        if detection_result.face_blendshapes == []:
            continue
        else:
            counter += 1
        
        # Introduce a small delay to prevent processor overload, especially during extensive loops
        if counter % 500 == 0:
            time.sleep(0)  # Sleep for 0 seconds (effectively yields control)
        
        # Compare each detected blendshape's score with the threshold
        for blendshape_category in detection_result.face_blendshapes[0]:
            if blendshape_category.score > 0.4:
                # Increment the count for the blendshape if its score is above the threshold
                blend_shapes[str(blendshape_category.index)] = (
                    blend_shapes[str(blendshape_category.index)] + 1
                )

    return blend_shapes


# List of blendshape indices identified as most relevant from previous analysis.
# This list is typically derived manually after running `choosing_blendshapes`
# and analyzing its output to select blendshapes highly correlated with specific emotions.
blends_to_print = [
    "1", "2", "3", "4", "5", "9", "10", "11", "12", "13", "14", "15",
    "16", "17", "18", "19", "20", "25", "34", "35", "38", "44", "45",
    "46", "47", "48", "49", "emotion", # 'emotion' is a placeholder for the label, not a blendshape index
]


# Example usage when run as a script:
# if __name__ == "__main__":
#     # Load a balanced dataset (e.g., FER2013) for analysis
#     dataset = balanced_dataset(os.getenv("FER2013"))
#     # Determine relevant blendshapes from the dataset
#     blendshapes = choosing_blendshapes(dataset)
#     print("Relevant Blendshapes and their counts:", blendshapes)
