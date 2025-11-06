"""Module for creating a blendshapes dataset from image data.

This script processes an image dataset, extracts facial blendshapes using MediaPipe,
and then generates a CSV file containing these blendshapes along with corresponding
emotion labels. It focuses on a subset of important blendshapes as identified
by previous research.
"""

import csv
import numpy as np
import tensorflow as tf
import mediapipe as mp
from mediapipe_tools.visualizing_and_setup import detector
from utils.csv_writer import csv_writer
from dotenv import load_dotenv
import os
import sys

load_dotenv()

# Initialize lists to store dataset components
set = []        # Stores raw data from the input CSV
images = []     # Stores processed image data (NumPy arrays)
labels = []     # Stores emotion labels
full_set = []   # Not used in current script, but kept for context if needed
indices = []    # Not used in current script, but kept for context if needed


# Load the dataset to be processed (e.g., validation set)
with open(os.getenv("VAL_DATASET"),
    mode="r",
) as data:
    csvFile = csv.reader(data)
    next(csvFile)  # Skip header row
    for lines in csvFile:
        set.append(lines)
    
    # Iterate over each image's pixels in the dataset to convert into an RGB image
    for i in range(len(set) - 1):
        # Convert pixel string to NumPy array and reshape to 48x48 grayscale
        image = np.array(set[i][1].split()).reshape(48, 48, 1).astype(np.uint8)
        
        # Convert to TensorFlow tensor, make proto, make ndarray, then convert to RGB
        # These steps are necessary for MediaPipe compatibility.
        image = tf.convert_to_tensor(image)
        image = tf.make_tensor_proto(image, dtype=tf.uint8)
        image = tf.make_ndarray(image).reshape(48, 48, 1)
        image = tf.convert_to_tensor(image, dtype=tf.uint8)
        image = tf.image.grayscale_to_rgb(image).numpy()
        
        images.append(image)
        labels.append(int(set[i][0]))
        # indices.append(int(set[i][2])) # Uncomment if indices are needed

# Define the important blendshapes based on ablation studies from the research paper
blends_to_print = [
    "1", "2", "3", "4", "5", "9", "10", "11", "12", "13", "14", "15",
    "16", "17", "18", "19", "20", "25", "34", "35", "38", "44", "45",
    "46", "47", "48", "49", "emotion", # 'emotion' is a placeholder for the label
]

# Initialize a NumPy array to store blendshape scores and the emotion label
# The size 36 is based on 35 blendshapes + 1 for the emotion label.
arr = np.zeros((len(images), 36))

# Process each image to detect blendshapes and populate the array
for ele in range(len(images) - 1):
    img = images[ele]
    label = labels[ele]
    # img_index = indices[ele] # Uncomment if image indices are needed
    
    # Create MediaPipe Image object
    frame = mp.Image(image_format=mp.ImageFormat.SRGB, data=img)
    
    # Detect face blendshapes using MediaPipe detector
    face_landmarker_detector = detector()
    detection_result = face_landmarker_detector.detect(frame)
    
    cat_counter = 0
    # If blendshapes are detected, extract and store the scores for selected blendshapes
    if detection_result.face_blendshapes != []:
        print(detection_result.face_blendshapes[0]) # Debugging print
        for category in detection_result.face_blendshapes[0]:
            if str(category.index) in blends_to_print:
                arr[ele, cat_counter] = category.score
                cat_counter += 1
            else:
                continue
        arr[ele, 35] = label  # Store the emotion label in the last column
    # arr[ele, 35] = img_index # Uncomment if image indices are to be stored instead of label


# Define fields for the output CSV file
fields = ["emotion", "pixels", "Index"] # Note: 'pixels' and 'Index' might not be directly used if 'arr' is written.

# Write the extracted blendshapes and labels to a CSV file
# The 'beedoo' argument is a placeholder and might need to be replaced with actual header if 'fields' is not used.
csv_writer("blends_val_all_emotion.csv", "beedoo", arr)
