#!/usr/bin/env python3

"""
Visualizing images using MediaPipe and plot the landmarks on them
then plot the bar graph for the blendshape values for the plotted 
image
"""

import matplotlib.pyplot as plt
import tensorflow as tf
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2


def draw_landmarks_on_image(rgb_image, detection_result):
    """detect and draw landmarks on a single image for more info: 
    https://colab.research.google.com/github/googlesamples/mediapipe/blob/main/examples/face_landmarker/python/%5BMediaPipe_Python_Tasks%5D_Face_Landmarker.ipynb"""
    
    face_landmarks_list = detection_result.face_landmarks
    annotated_image = tf.keras.ops.copy(rgb_image).numpy()

    # Loop through the detected faces to visualize.
    for idx in range(len(face_landmarks_list)):
        face_landmarks = face_landmarks_list[idx]

      # Draw the face landmarks.
        face_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        face_landmarks_proto.landmark.extend([
      landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in face_landmarks
      ])

        solutions.drawing_utils.draw_landmarks(
          image=annotated_image,
          landmark_list=face_landmarks_proto,
          connections=mp.solutions.face_mesh.FACEMESH_TESSELATION,
          landmark_drawing_spec=None,
          connection_drawing_spec=mp.solutions.drawing_styles
          .get_default_face_mesh_tesselation_style())
        solutions.drawing_utils.draw_landmarks(
          image=annotated_image,
          landmark_list=face_landmarks_proto,
          connections=mp.solutions.face_mesh.FACEMESH_CONTOURS,
          landmark_drawing_spec=None,
          connection_drawing_spec=mp.solutions.drawing_styles
          .get_default_face_mesh_contours_style())
        solutions.drawing_utils.draw_landmarks(
          image=annotated_image,
          landmark_list=face_landmarks_proto,
          connections=mp.solutions.face_mesh.FACEMESH_IRISES,
          landmark_drawing_spec=None,
          connection_drawing_spec=mp.solutions.drawing_styles
          .get_default_face_mesh_iris_connections_style())

    return annotated_image


def plot_face_blendshapes_bar_graph(face_blendshapes):
    """plot the blendshapes histogam for the sample image"""
    # Extract the face blendshapes category names and scores.
    face_blendshapes_names = [
      face_blendshapes_category.category_name for face_blendshapes_category in face_blendshapes
      ]
    face_blendshapes_scores = [
      face_blendshapes_category.score for face_blendshapes_category in face_blendshapes
      ]
    # The blendshapes are ordered in decreasing score value.
    face_blendshapes_ranks = range(len(face_blendshapes_names))

    fig, ax = plt.subplots(figsize=(12, 12))
    bar = ax.barh(
      face_blendshapes_ranks, face_blendshapes_scores, label=[str(x) for x in face_blendshapes_ranks]
      )
    ax.set_yticks(face_blendshapes_ranks, face_blendshapes_names)
    ax.invert_yaxis()

    # Label each bar with values
    for score, patch in zip(face_blendshapes_scores, bar.patches):
        plt.text(patch.get_x() + patch.get_width(), patch.get_y(), f"{score:.4f}", va="top")

    ax.set_xlabel('Score')
    ax.set_title("Face Blendshapes")
    plt.tight_layout()
    plt.show()


def detector():
    # Create an FaceLandmarker object.
    base_options = python.BaseOptions(
      model_asset_path='/home/samer/Desktop/HAN stuff/Big data Small Data/BDSD/Minor_project/BDSD_Minor_Project/face_landmarker_v2_with_blendshapes.task',
      delegate=mp.tasks.BaseOptions.Delegate.GPU
      )
    options = vision.FaceLandmarkerOptions(base_options=base_options,
                                          output_face_blendshapes=True,
                                          output_facial_transformation_matrixes=True,
                                          num_faces=1)
    detector = vision.FaceLandmarker.create_from_options(options)
    
    return detector


def detect_and_graph():
    """
    detect the face and the landmarks in the sample image and display the 
    annotated image and the blendshapes histogram
    """
    # Load the input image.
    image = mp.Image.create_from_file("image.png")

    # Detect face landmarks from the input image.
    detection_result = detector()
    detection_result = detection_result.detect(image)
    # print(detection_result.facial_transformation_matrixes)

    # Process the detection result. In this case, visualize it.
    annotated_image = draw_landmarks_on_image(image.numpy_view(), detection_result)
    plt.imshow(annotated_image)
    plt.show()

    plot_face_blendshapes_bar_graph(detection_result.face_blendshapes[0])
    # print(detection_result.face_blendshapes[0])
    print(detection_result)



# if __name__ == "__main__":
#     detect_and_graph()
