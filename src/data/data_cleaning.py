"""clean the dataset from images non-recognizable by mediapipe
by checking for the mediapipe detector output if it does not exists
the image get deleted"""

import sys
sys.path.insert(0,"/home/samer/Desktop/HAN stuff/Big data Small Data/BDSD/Minor_project/BDSD_Minor_Project/src")
import tensorflow as tf
import mediapipe as mp
from mediapipe_tools.visualizing_and_setup import detector
from data_processing import balanced_dataset
from utils.csv_writer import csv_writer


# creat lists for dataset splits
def list_creator(fullset):
    """create the three sets lists from the full samples list"""
    training_set = []
    validation_set = []
    test_set = []
    for i in fullset:
        if i[2] == 'Training':
            training_set.append(i)
        elif i[2] == 'PublicTest':
            validation_set.append(i)
        elif i[2] == 'PrivateTest':
            test_set.append(i)

    return training_set, validation_set, test_set


def sets_cleaner(training_set, validation_set, test_set):
    """create three dataset with all images understandable by mediapipe"""
    training_set_hus = []
    validation_set_hus = []
    test_set_hus = []

    # append the understandable images of the original dataset splits to new lists
    for lines in training_set:
        image = str(lines[1]).split(" ")
        print(image)
        image = tf.convert_to_tensor(image)
        image = tf.make_tensor_proto(image, dtype = tf.uint8)
        image = tf.make_ndarray(image).reshape(48, 48, 1)
        image = tf.convert_to_tensor(image, dtype=tf.uint8)
        image = tf.image.grayscale_to_rgb(image).numpy()
      #   plt.imshow(image)
      #   plt.show()
        frame = mp.Image(image_format=mp.ImageFormat.SRGB,data=image)
        detection_result = detector()
        detection_result = detection_result.detect(frame)
        if detection_result.face_blendshapes == []:
          continue
        else:
          training_set_hus.append(lines)

    for val_line in validation_set:
        imagee = str(tf.strings.as_string(val_line).numpy().decode("utf-8")).split(",")[1].split(" ")
        imagee = tf.convert_to_tensor(imagee)
        imagee = tf.make_tensor_proto(imagee, dtype = tf.uint8)
        imagee = tf.make_ndarray(imagee).reshape(48, 48, 1)
        imagee = tf.convert_to_tensor(imagee, dtype=tf.uint8)
        imagee = tf.image.grayscale_to_rgb(imagee).numpy()
      #   plt.imshow(image)
      #   plt.show()
        framee = mp.Image(image_format=mp.ImageFormat.SRGB,data=imagee)
        detection_result = detector()
        detection_result = detection_result.detect(framee)
        if detection_result.face_blendshapes == []:
          continue
        else:
          validation_set_hus.append(val_line)

    for test_lines in test_set:
        imageee = str(tf.strings.as_string(test_lines).numpy().decode("utf-8")).split(",")[1].split(" ")
        imageee = tf.convert_to_tensor(imageee)
        imageee = tf.make_tensor_proto(imageee, dtype = tf.uint8)
        imageee = tf.make_ndarray(imageee).reshape(48, 48, 1)
        imageee = tf.convert_to_tensor(imageee, dtype=tf.uint8)
        imageee = tf.image.grayscale_to_rgb(imageee).numpy()
      #   plt.imshow(image)
      #   plt.show()
        frameee = mp.Image(image_format=mp.ImageFormat.SRGB,data=imageee)
        detection_result = detector()
        detection_result = detection_result.detect(frameee)
        if detection_result.face_blendshapes == []:
          continue
        else:
          test_set_hus.append(test_lines)

    return  training_set_hus, validation_set_hus, test_set_hus


def write_files(training_set_hus, validation_set_hus, test_set_hus):
    """create files for the new dataset splits for the created lists"""
    fields = ["emotion", "pixels", "Usage"]
    csv_writer("training_set_full.csv", fields, training_set_hus)
    csv_writer("validation_set_full.csv", fields, validation_set_hus)
    csv_writer("test_set_full.csv", fields, test_set_hus)




if __name__=="__main__":
    fullset = balanced_dataset("/home/samer/Desktop/HAN stuff/Big data Small Data/BDSD/Minor_project/BDSD_Minor_Project/Datasets/training_set_full.csv")
    training_set, validation_set, test_set = list_creator(fullset)
    training_set_hus, validation_set_hus, test_set_hus = sets_cleaner(training_set, validation_set, test_set)
    # write_files(training_set_hus, validation_set_hus, test_set_hus)
    