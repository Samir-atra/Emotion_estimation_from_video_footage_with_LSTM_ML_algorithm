# Emotion estimation from camera footage with LSTM

The code for the research paper: [Emotion estimation from video footage with LSTM](https://www.researchgate.net/publication/387438191_EMOTION_ESTIMATION_FROM_VIDEO_FOOTAGE_WITH_LSTM)

Blendshapes dataset mentioned in the paper: [Blendshapes dataset](https://doi.org/10.34740/KAGGLE/DSV/10716347)

## Introduction

To use this repository start by:
- Cloning the repository on the local machine
- Creating the conda environment in 

```bash
environment.yml
```

Then
- Running the commands in the script one at a time in the command line
```bash
wget -O face_landmarker_v2_with_blendshapes.task -q https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1face_landmarker.task
```
To download the mediapipe task for the face landmark detection

```bash
wget -q -O image.png https://storage.googleapis.com/mediapipe-assets/business-person.png
```
To download a sample image to testing and visualizing the blendshapes

```bash
export PYTHONPATH=$PYTHONPATH:"<path to the src directory on the machine>"
```
Adding the code location to the python path.

- Download the dataset from https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data
- Then process the dataset and load it train the model then evaluate.

## Reference:

Samer Attrah. (2025). FER2013 blendshapes dataset example (Partial) [Data set]. Kaggle. https://doi.org/10.34740/KAGGLE/DSV/10716347
