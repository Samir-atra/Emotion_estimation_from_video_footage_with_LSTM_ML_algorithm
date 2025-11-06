"""Module for generating predictions and measuring inference latency of an LSTM model.

This module provides a function to load a pre-trained Keras model, use it to
generate predictions on a test set, and calculate the average inference latency
per prediction.
"""

import numpy as np
import keras
import time
from dotenv import load_dotenv
import os 
import sys
import csv

load_dotenv()

def generate_prediction(x_test):
    """
    Uses a saved Keras model to generate predictions for the test set
    and measures the inference latency.

    This function loads a pre-trained model, reads the test dataset from a CSV file,
    preprocesses the data, and then iterates through each instance to make a prediction.
    It records the time taken for each prediction to calculate average latency.

    Args:
        x_test (list): A placeholder argument. The actual test data is loaded internally
                       from the CSV file specified by the `TEST_DATASET` environment variable.

    Returns:
        list: A list of predicted classes (integers) for each instance in the test set.
    """

    # Load the pre-trained Keras model from the path specified in environment variables
    model = keras.models.load_model(os.getenv("SAVED_MODEL_PATH"))
    model.summary() # Print a summary of the loaded model's architecture
    
    predictions = [] # List to store predicted class labels
    latencies = []   # List to store inference latencies for each prediction
    all_data = []    # List to store all data read from the test CSV
    
    # Read the test dataset from the CSV file
    with open(os.getenv("TEST_DATASET"), mode="r", encoding="utf-8") as data_file:
        csv_reader = csv.reader(data_file)
        next(csv_reader)  # Skip the header row
        for row in csv_reader:
            all_data.append(row)

    # Convert the raw data to a NumPy array for easier slicing and type conversion
    data_array = np.array(all_data, dtype=np.float64)
    # Extract blendshape features (first 52 columns) from the data array
    x_test_processed = data_array[:, 0:52]
    
    # Iterate through each instance in the processed test data to make predictions
    for inst in x_test_processed:
        start_time = time.perf_counter() # Record start time for latency measurement
        
        # Convert instance to NumPy array and reshape for LSTM input (1, timesteps, features)
        inst = np.array(inst, dtype=np.float64)
        inst = np.reshape(inst, (1, 52, 1))

        # Make a prediction using the loaded model (verbose=0 suppresses output)
        y_pred = model.predict(inst, verbose=0)
        y_pred = np.argmax(y_pred) # Get the class with the highest probability
        end_time = time.perf_counter()   # Record end time
        
        predictions.append(y_pred) # Store the predicted class
        latency_ms = (end_time - start_time) * 1000 # Calculate latency in milliseconds
        latencies.append(latency_ms) # Store the latency
    
    avg_latency = np.mean(latencies) # Calculate the average latency
    print(f"Average latency: {avg_latency:.2f} ms") # Print average latency
    return predictions

if __name__ == "__main__":
    # Example usage when the script is executed directly.
    # An empty string is passed as x_test, as the actual data is loaded internally.
    generate_prediction("")
    