"""choosing relevant blendshapes for the happy, sad dataset and the 
happy, sad, neutral dataset""" 

import time
import cv2
import numpy as np
import mediapipe as mp


def choosing_blendshapes(training_set_hus):
    # create a dictionary with numbers from 0 to 52 
    blend_shapes = dict()
    for i in range(0,52):
        blend_shapes[str(i)] = 0

    print(blend_shapes)
    # sad = 0
    # happy = 0
    counter = 0

    # find which blendshapes are most relevant to happiness and sadness by passing them on a 0.4 threshold 
    for i in range(len(training_set_hus)):
        image = np.array(training_set_hus[i][1].split(' ')).reshape(48, 48, 1).astype(np.uint8)    # build images from pixels lists
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        frame = mp.Image(image_format=mp.ImageFormat.SRGB,data=image)
        detection_result = detector.detect(frame)
        if detection_result.face_blendshapes == []:
            continue
        else:
            counter += 1              
        if counter %500 == 0:
            time.sleep(5)                 # sleep 5 seconds to avoid processor overload
        for i in detection_result.face_blendshapes[0]:           # compare each blendshape in for the current image with the threshold
            if i.score > 0.4:
                blend_shapes[str(i.index)] = blend_shapes[str(i.index)] + 1           # edit the counts dictionary
    print(blend_shapes)

    return blend_shapes
# from the resulting dictionary, manually find the modst relevent blendshapes and insert them in a list

blends_to_print = ['1', '2', '3', '4', '5', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '25', '34', '35', '38', '44', '45', '46', '47', '48', '49', 'emotion'] # list of blendshapes indices that are most relevent
