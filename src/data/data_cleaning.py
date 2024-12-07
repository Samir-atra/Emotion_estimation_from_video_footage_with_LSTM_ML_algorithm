"""clean the dataset from images non-recognizable by mediapipe
by checking for the mediapipe detector output if it does not exists
the image get deleted"""

import numpy as np
import csv
import cv2

training_set = []
validation_set = []
test_set = []

# creat lists for dataset splits
for i in fullset:
    if i[2] == 'Training':
        training_set.append(i)
    elif i[2] == 'PublicTest':
        validation_set.append(i)
    elif i[2] == 'PrivateTest':
        test_set.append(i)

training_set_hus = []
validation_set_hus = []
test_set_hus = []

# append the understandable images of the original dataset splits to new lists
for lines in training_set:
      image = np.array(lines[1].split(' ')).reshape(48, 48, 1).astype(np.uint8)
      image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    #   plt.imshow(image)
    #   plt.show()
      frame = mp.Image(image_format=mp.ImageFormat.SRGB,data=image)
      detection_result = detector.detect(frame)
      if detection_result.face_blendshapes == []:
        continue
      else:
        training_set_hus.append(lines)

for val_line in validation_set:
      imagee = np.array(val_line[1].split(' ')).reshape(48, 48, 1).astype(np.uint8)
      imagee = cv2.cvtColor(imagee, cv2.COLOR_GRAY2RGB)
    #   plt.imshow(image)
    #   plt.show()
      framee = mp.Image(image_format=mp.ImageFormat.SRGB,data=imagee)
      detection_result = detector.detect(framee)
      if detection_result.face_blendshapes == []:
        continue
      else:
        validation_set_hus.append(val_line)

for test_lines in test_set:
      imageee = np.array(test_lines[1].split(' ')).reshape(48, 48, 1).astype(np.uint8)
      imageee = cv2.cvtColor(imageee, cv2.COLOR_GRAY2RGB)
    #   plt.imshow(image)
    #   plt.show()
      frameee = mp.Image(image_format=mp.ImageFormat.SRGB,data=imageee)
      detection_result = detector.detect(frameee)
      if detection_result.face_blendshapes == []:
        continue
      else:
        test_set_hus.append(test_lines)

# create files for the new dataset splits for the created lists
fields = ["emotion", "pixels", "Usage"]
csv_writer("training_set_full.csv", fields, training_set_hus)
csv_writer("validation_set_full.csv", fields, validation_set_hus)
csv_writer("test_set_full.csv", fields, test_set_hus)
