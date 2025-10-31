"""pridict the class of a single image"""

import numpy as np
import keras
import time
from dotenv import load_dotenv
import os 
import sys
import csv

load_dotenv()

def generate_prediction(x_test):
    """use a saved model to generate predictions to the test set

    Args:
        x_test (list): the blendshapes of the test set

    Returns:
        predictions (list): the predicted classes of the test set
    """

    model = keras.models.load_model(os.getenv("SAVED_MODEL_PATH"))   # load the model to be used for prediction
    model.summary()
    predictions = []
    latencies = []
    all_data = []
    with open(os.getenv("TEST_DATASET"), mode="r", encoding="utf-8") as data_file:
        csv_reader = csv.reader(data_file)
        next(csv_reader)  # Skip header row
        for row in csv_reader:
            all_data.append(row)

    # Convert to numpy array for easier slicing
    data_array = np.array(all_data, dtype=np.float64)
    x_test = data_array[:, 0:52]
    for inst in x_test:
        start_time = time.perf_counter()
        inst = np.array(inst, dtype=np.float64)
        inst = np.reshape(inst, (1, 52, 1))     

        y_pred = model.predict(inst, verbose=0)   # predict the labels for the images
        y_pred = np.argmax(y_pred)
        end_time = time.perf_counter()
        predictions.append(y_pred)
        latency_ms = (end_time - start_time) * 1000
        latencies.append(latency_ms)
    avg_latency = np.mean(latencies)
    print(f"Average latency: {avg_latency:.2f} ms")
    return predictions

if __name__ == "__main__":
    generate_prediction("")
    