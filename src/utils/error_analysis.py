"""Module for analyzing and visualizing errors in emotion estimation.

This module provides functions to:
- Identify and visualize misclassified images from a test set.
- Create class-specific datasets and visualize blendshape correlations.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import csv
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
import mediapipe as mp
import numpy as np
from dotenv import load_dotenv

# Local application imports
from mediapipe_tools.visualizing_and_setup import detector, draw_landmarks_on_image
from utils.csv_writer import csv_writer
from utils.prediction_and_latency import generate_prediction


load_dotenv()

def count_visualize_errors(x_test):
    """
    Counts misclassifications by comparing model predictions against ground truth labels
    and visualizes the misclassified images with MediaPipe landmarks.

    This function performs the following steps:
    1. Generates predictions for the provided test set (`x_test`).
    2. Reads ground truth labels and image data from a CSV file specified by `FULL_TEST_SET`
       in the environment variables.
    3. Identifies images where the prediction does not match the ground truth.
    4. For each misclassified image:
        - Converts the image data from a string representation to a NumPy array.
        - Converts the grayscale image to RGB format for MediaPipe processing.
        - Detects facial landmarks using the MediaPipe FaceLandmarker.
        - Draws the detected landmarks on the image.
        - Saves and displays both the original and annotated misclassified images.

    Args:
        x_test (list): A list of input features (e.g., blendshapes) for the test set.
                       This is passed to `generate_prediction` to get model outputs.
    """

    # Create directories for storing original and annotated misclassified images
    original_dir = "original_images"
    annotated_dir = "annotated_images"
    os.makedirs(original_dir, exist_ok=True)
    os.makedirs(annotated_dir, exist_ok=True)

    # Generate predictions using the trained model
    predictions = generate_prediction(x_test)
    ground_truth = []
    errors = []
    test_img_set = []
    test_index_set = []

    # Read ground truth labels and image data from the full test set CSV
    with open(os.getenv("FULL_TEST_SET"), mode="r", encoding="utf-8") as test_img_data:
        csv_img_file = csv.reader(test_img_data)
        next(csv_img_file)  # Skip the header row

        # Populate ground truth labels, image data, and test indices
        for lines in csv_img_file:
            ground_truth.append(int(lines[0]))  # Emotion label
            test_img_set.append(lines[1])      # Image data (as string)
            test_index_set.append(lines[2])    # Original index of the image

        # Compare predictions with ground truth to find errors
        for j in range(len(predictions)):
            if predictions[j] != ground_truth[j]:
                errors.append(test_index_set[j])  # Store index of misclassified images

        # Visualize each misclassified image
        for i, ind in enumerate(errors):
            # Convert image string to numpy array and reshape to 48x48 grayscale
            imageee = (
                np.array(str(test_img_set[int(float(ind))]).split(" "))
                .reshape(48, 48, 1)
                .astype(np.uint8)
            )
            # Convert grayscale image to RGB for MediaPipe processing
            imageees = cv2.cvtColor(imageee, cv2.COLOR_GRAY2RGB)
            
            # Create MediaPipe Image object
            frame = mp.Image(image_format=mp.ImageFormat.SRGB, data=imageees)
            
            # Detect facial landmarks
            face_landmarker_detector = detector()
            detection_result = face_landmarker_detector.detect(frame)
            
            # Draw landmarks on the image
            annotated_image = draw_landmarks_on_image(
                frame.numpy_view(), detection_result
            )
            print("Annotated image generated.")

            # Save and display the annotated image
            plt.figure()
            plt.imshow(annotated_image)
            annotated_filename = os.path.join(annotated_dir, f"annotated_image_{i}.png")
            plt.savefig(annotated_filename)
            plt.show()  # Display the image
            # os.system(f'xdg-open {annotated_filename}')
            
            # Save and display the original image
            plt.figure()
            plt.imshow(imageees)
            original_filename = os.path.join(original_dir, f"original_image_{i}.png")
            plt.savefig(original_filename)
            plt.show()  # Display the image
            # os.system(f'xdg-open {original_filename}')

def create_classes_corrcoef(traindata):
    """
    Organizes training data into class-specific lists, writes them to CSV files,
    and visualizes blendshape distributions and correlation matrices.

    This function performs the following steps:
    1. Divides the `traindata` into three lists (`class1`, `class2`, `class3`)
       based on the emotion label (assumed to be at index 52).
    2. Writes each class list to a separate CSV file using `csv_writer`.
    3. Generates a multi-subplot figure to visualize the distribution of 51 blendshape
       values across the first 2000 instances of each class.
    4. Calculates and prints the correlation coefficient matrix between `class1_slice`
       and `class2_slice` for a specific blendshape.

    Args:
        traindata (list): A list of training set instances, where each instance
                          is expected to be a list or array-like object with
                          blendshape values and an emotion label.
    """

    class1 = []
    class2 = []
    class3 = []

    # Categorize training instances into respective class lists
    for ins in traindata:
        if float(ins[52]) == 0:
            class1.append(ins)
        elif float(ins[52]) == 1:
            class2.append(ins)
        elif float(ins[52]) == 2:
            class3.append(ins)

    # Write class-specific data to CSV files
    csv_writer("class1.csv", "beedoo", class1)
    csv_writer("class2.csv", "beedoo", class2)
    csv_writer("class3.csv", "beedoo", class3)

    # Prepare for blendshape visualization
    plt.figure(figsize=[40, 40])
    y = np.linspace(0, 1999, 1999)    # X-axis values for plotting blendshape distributions
    
    # Plot blendshape distributions for each class
    for j in range(1, 52):
        plt.subplot(10, 6, j)         # Create subplots for each blendshape
        class1_slice = [float(i[j]) for i in class1[1:2000]]
        class2_slice = [float(i[j]) for i in class2[1:2000]]
        class3_slice = [float(i[j]) for i in class3[1:2000]]
        plt.title(f"Blendshape {j}")
        plt.scatter(y, class1_slice, color="blue", label="Class 0")
        plt.scatter(y, class2_slice, color="red", label="Class 1")
        plt.scatter(y, class3_slice, color="green", label="Class 2")
        plt.legend()

    print(class1[1:100]) # Print a slice of class1 data for inspection
    
    # Calculate and print the correlation coefficient matrix
    corr_mat = np.corrcoef(class1_slice, class2_slice)
    print("Correlation matrix between Class 0 and Class 1 blendshape values:", corr_mat)


if __name__ == "__main__":
    # Example usage: Run error visualization with a placeholder for x_test
    # In a real scenario, x_test would be populated with actual test data.
    count_visualize_errors("")
    