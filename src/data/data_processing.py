"""
count the number of images in each class and the images ignored from 
the dataset and create class balanced dataset
"""

import csv
import mediapipe as mp
import matplotlib.pyplot as plt
import tensorflow as tf
from mediapipe_tools.visualizing_and_setup import detector


def categories_and_unreadable_counter(fer2013_path):
    """
    counts the number of images in each class in one dictionary and
    the number of the images unreadable by mediapipe in another dictionary
    """
    categories_counts = {"0": 0, "1": 0, "2": 0, "3": 0, "4": 0, "5": 0, "6": 0}
    skipped = {"0": 0, "1": 0, "2": 0, "3": 0, "4": 0, "5": 0, "6": 0}

    data = tf.io.read_file(fer2013_path)  # open the dataset file
    f = tf.strings.split(data, sep="\n")
    for lines in f[1:-1]:  # loop throught the training instances
        print(type(lines))
        key = str(tf.strings.as_string(lines).numpy().decode("utf-8")).split(",")[0]
        if key in categories_counts.keys():
            categories_counts[key] = categories_counts[key] + 1
        image = (
            str(tf.strings.as_string(lines).numpy().decode("utf-8"))
            .split(",")[1]
            .split(" ")
        )
        image = tf.convert_to_tensor(image)
        image = tf.make_tensor_proto(image, dtype=tf.uint8)
        image = tf.make_ndarray(image).reshape(48, 48, 1)
        image = tf.convert_to_tensor(image, dtype=tf.uint8)
        image = tf.image.grayscale_to_rgb(image).numpy()
        rgb_frame = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
        detection_result = detector()
        detection_result = detection_result.detect(
            rgb_frame
        )  # detect the face in the image
        # check if the image get recognized or skipped by mediapipe
        if (
            detection_result.face_blendshapes == []
        ):  # check if mediapipe is able to detect a face in the image
            skipped[key] = skipped[key] + 1
        else:
            continue
    print(categories_counts)
    print(skipped)


def balanced_dataset(full_set_path):
    """
    Create a dataset with equal number of images for each
    class of the eight in the original dataset and create from them a dataset
    of three classes Happy, sad and unknown
    """

    fullset = []
    Training_set = []
    validation_set = []
    test_set = []
    class_counter = {"0": 0, "1": 0, "2": 0, "3": 0, "4": 0, "5": 0, "6": 0}

    with open(full_set_path, mode="r", encoding="utf-8") as data:
        csvfile = csv.reader(data)
        next(csvfile)
        # iterate over the dataset and append the images to a list, for limited numbers according to it's classes, to creat the training set
        for lines in csvfile:
            if lines[0] == "0" and class_counter["0"] < 1500:
                class_counter[lines[0]] = class_counter[lines[0]] + 1
                lines[0] = "1"
                fullset.append(lines)
            elif lines[0] == "1" and class_counter["1"] < 1500:
                class_counter[lines[0]] = class_counter[lines[0]] + 1
                lines[0] = "1"
                fullset.append(lines)
            elif lines[0] == "2" and class_counter["2"] < 1500:
                class_counter[lines[0]] = class_counter[lines[0]] + 1
                lines[0] = "1"
                fullset.append(lines)
            elif lines[0] == "3" and class_counter["3"] < 4000:
                class_counter[lines[0]] = class_counter[lines[0]] + 1
                lines[0] = "0"
                fullset.append(lines)
            elif lines[0] == "4" and class_counter["4"] < 4000:
                class_counter[lines[0]] = class_counter[lines[0]] + 1
                lines[0] = "2"
                fullset.append(lines)
            elif lines[0] == "5" and class_counter["5"] < 1500:
                class_counter[lines[0]] = class_counter[lines[0]] + 1
                lines[0] = "1"
                fullset.append(lines)
            elif lines[0] == "6" and class_counter["6"] < 1500:
                class_counter[lines[0]] = class_counter[lines[0]] + 1
                lines[0] = "1"
                fullset.append(lines)

    print(class_counter)
    return fullset


# if __name__ == "__main__":
#     categories_and_unreadable_counter("/home/samer/Desktop/HAN stuff/Big data Small Data/BDSD/Minor_project/BDSD_Minor_Project/Datasets/fer2013.csv")
#     balanced_dataset("/home/samer/Desktop/HAN stuff/Big data Small Data/BDSD/Minor_project/BDSD_Minor_Project/Datasets/training_set_full.csv")
