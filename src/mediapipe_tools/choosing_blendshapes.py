"""
choosing relevant blendshapes for the happy, sad dataset and the 
happy, sad, neutral dataset
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

# configure the gpu to use a bigger portion of the memory for processing the data
gpus = tf.config.list_physical_devices("GPU")
tf.config.experimental.set_synchronous_execution(True)
if gpus:
    try:
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
    Count the occurances of each blendshape in the dataset images and when
    compared to a threshold

    Args:
        training_set_hus (list): training dataset as a python list.

    Returns:
        blend_shapes (dict): a dictionary of the blendshapes and thier counts.
    """

    blend_shapes = {}
    for i in range(0, 52):
        blend_shapes[str(i)] = 0

    counter = 0

    # find which blendshapes are most relevant to happiness and sadness by passing them on a 0.4 threshold
    for i in range(len(training_set_hus)):
        image = str(training_set_hus[i][1]).split(" ")
        image = tf.convert_to_tensor(image)
        # serialize the tensor to be oricessed into a numpy array and reshaped
        image = tf.make_tensor_proto(image, dtype=tf.uint8)
        image = tf.make_ndarray(image).reshape(48, 48, 1)
        image = tf.convert_to_tensor(image, dtype=tf.uint8)
        image = tf.image.grayscale_to_rgb(image).numpy()
        frame = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
        detection_result = detector()
        detection_result = detection_result.detect(frame)
        if detection_result.face_blendshapes == []:
            continue
        else:
            counter += 1
        if counter % 500 == 0:
            time.sleep(0)  # sleep 5 seconds to avoid processor overload
        for i in detection_result.face_blendshapes[
            0
        ]:  # compare each blendshape in for the current image with the threshold
            if i.score > 0.4:
                blend_shapes[str(i.index)] = (
                    blend_shapes[str(i.index)] + 1
                )  # edit the counts dictionary

    return blend_shapes


# from the resulting dictionary, manually find the modst relevent blendshapes and insert them in a list

blends_to_print = [
    "1",
    "2",
    "3",
    "4",
    "5",
    "9",
    "10",
    "11",
    "12",
    "13",
    "14",
    "15",
    "16",
    "17",
    "18",
    "19",
    "20",
    "25",
    "34",
    "35",
    "38",
    "44",
    "45",
    "46",
    "47",
    "48",
    "49",
    "emotion",
]  # list of blendshapes indices that are most relevent


# if __name__ == "__main__":
#     dataset = balanced_dataset(os.getenv("FER2013")
#     )
#     blendshapes = choosing_blendshapes(dataset)
#     print(blendshapes)
