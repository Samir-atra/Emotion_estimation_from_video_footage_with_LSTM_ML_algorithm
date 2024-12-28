"""
create the blendshapes dataset for the images dataset,
by extracting the blendshapes of each image and creating 
a .csv file to  
"""

import csv
import numpy as np
import tensorflow as tf
import mediapipe as mp
from mediapipe_tools.visualizing_and_setup import detector
from utils.csv_writer import csv_writer


set = []
images = []
labels = []
full_set = []
indices = []


with open(
    "/home/samer/Desktop/HAN stuff/Big data Small Data/BDSD/Minor_project/BDSD_Minor_Project/Datasets/validation_set_full.csv",
    mode="r",
) as data:  # load the dataset that will be processed
    csvFile = csv.reader(data)
    next(csvFile)
    for lines in csvFile:
        set.append(lines)
    for i in range(len(set) - 1):
        image = np.array(set[i][1].split()).reshape(48, 48, 1).astype(np.uint8)
        image = tf.convert_to_tensor(image)
        image = tf.make_tensor_proto(image, dtype=tf.uint8)
        image = tf.make_ndarray(image).reshape(48, 48, 1)
        image = tf.convert_to_tensor(image, dtype=tf.uint8)
        image = tf.image.grayscale_to_rgb(image).numpy()
        images.append(image)
        labels.append(int(set[i][0]))
        # indices.append(int(set[i][2]))


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
]


arr = np.zeros((len(images), 36))

# arr[0,:]= blendS_to_print
for ele in range(len(images) - 1):
    img = images[ele]
    label = labels[ele]
    # img_index = indices[ele]
    frame = mp.Image(image_format=mp.ImageFormat.SRGB, data=img)
    detection_result = detector()
    detection_result = detection_result.detect(frame)
    cat_counter = 0
    if detection_result.face_blendshapes != []:
        print(detection_result.face_blendshapes[0])
        for category in detection_result.face_blendshapes[0]:
            if str(category.index) in blends_to_print:
                arr[ele, cat_counter] = category.score
                cat_counter += 1
            else:
                continue
        arr[ele, 35] = label
    # arr[ele, 35] = img_index


fields = ["emotion", "pixels", "Index"]
csv_writer("blends_val_all_emotion.csv", "beedoo", arr)
