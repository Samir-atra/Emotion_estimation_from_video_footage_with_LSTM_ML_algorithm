"""clean the dataset from images non-recognizable by mediapipe
by checking for the mediapipe detector output if it does not exists
the image get deleted"""

import os
import tensorflow as tf
import mediapipe as mp
from dotenv import load_dotenv
from data_processing import balanced_dataset
from mediapipe_tools.visualizing_and_setup import detector
from utils.csv_writer import csv_writer

load_dotenv()

# creat lists for dataset splits
def list_creator(full_data_set):
    """create the three sets lists from the full samples list"""
    # initialize empty sets for the splits of te dataset
    training_set_list = []
    validation_set_list = []
    test_set_list = []
    for i in full_data_set:
        if i[2] == "Training":                # create the training split
            training_set_list.append(i)
        elif i[2] == "PublicTest":            # create the validation split
            validation_set_list.append(i)
        elif i[2] == "PrivateTest":
            test_set_list.append(i)                # create the test split


    return training_set_list, validation_set_list, test_set_list


def sets_cleaner(data_set_path):
    """create three dataset with all images understandable by mediapipe"""
    data_set = []

    data = tf.io.read_file(os.getenv(data_set_path)
    )  # open the dataset file
    f = tf.strings.split(data, sep="\n")
    for lines in f[1:-1]:  # loop throught the training instances
        image = (
            str(tf.strings.as_string(lines).numpy().decode("utf-8"))
            .split(",")[1]
            .split(" ")
        )
        # indirectly related steps to the process of checking the readablility of the images by
        # mediapipe, the steps are necessary and could not find a shorter way but they are time
        # but although tensorflow is much faster than numpy or other libraries which would have
        # needed one or two lines of code. for more: https://medium.com/@samiratra95/image-augmentation-using-tensorflow-and-mediapipe-baf54651f9fc
        image = tf.convert_to_tensor(image)
        image = tf.make_tensor_proto(image, dtype=tf.uint8)
        image = tf.make_ndarray(image).reshape(48, 48, 1)
        image = tf.convert_to_tensor(image, dtype=tf.uint8)
        image = tf.image.grayscale_to_rgb(image).numpy()
        #   plt.imshow(image)
        #   plt.show()
        frame = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
        detection_result = detector()
        detection_result = detection_result.detect(frame)
        if detection_result.face_blendshapes == []:
            continue
        else:
            data_set.append(str(tf.strings.as_string(lines).numpy().decode("utf-8")).split(","))

    return data_set


def write_files(training_data, validation_data, test_data):
    """create files for the new dataset splits for the created lists"""
    fields = ["emotion", "pixels", "Usage"]
    csv_writer("training_set_full.csv", fields, training_data)
    csv_writer("validation_set_full.csv", fields, validation_data)
    csv_writer("test_set_full.csv", fields, test_data)


if __name__=="__main__":
    fullset = balanced_dataset("/home/samer/Desktop/HAN stuff/Big data Small Data/BDSD/Minor_project/BDSD_Minor_Project/Datasets/training_set_full.csv")
    training_set, validation_set, test_set = list_creator(fullset)
    training_set_hus = sets_cleaner("TRAIN_DATASET")
    validation_set_hus = sets_cleaner("VAL_DATASET")
    test_set_hus = sets_cleaner("TEST_DATASET")
write_files(training_set_hus, validation_set_hus, test_set_hus)
