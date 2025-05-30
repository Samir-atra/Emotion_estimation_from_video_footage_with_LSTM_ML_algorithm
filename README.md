# Emotion estimation from camera footage with LSTM

## Author
Samer Attrah (Hogeschool van Arnhem en Nijmegen)

## Overview

The code in this repository is for the research paper: [Emotion estimation from video footage with LSTM](https://www.researchgate.net/publication/387438191_EMOTION_ESTIMATION_FROM_VIDEO_FOOTAGE_WITH_LSTM)

Blendshapes dataset mentioned in the paper: [Blendshapes dataset](https://doi.org/10.34740/KAGGLE/DSV/10716347)

## Introduction

To use this repository for replicating some of the processes described in the paper, start by:
- Cloning the repository on the local machine
- Creating the conda environment with suitable versions, where the program was built partially on Keras 2.X and other on Keras 3.X.

Then
- Running the commands in the script one at a time in the command line
```bash
wget -O face_landmarker_v2_with_blendshapes.task -q https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1face_landmarker.task
```
To download the mediapipe task for the face landmark detection

```bash
wget -q -O image.png https://storage.googleapis.com/mediapipe-assets/business-person.png
```
To download a sample image for testing and visualizing the blendshapes

- Download the FER2013 dataset from https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data
- Then process the dataset and load it, train the model then evaluate.

Or you can download the blendshapes dataset from: https://www.kaggle.com/datasets/samerattrah/fer2013-blendshapes-dataset-example-partial

and might use the pre-trained LSTM model, which you can find at: https://huggingface.co/SamerAttrah/Emotion_estimation_from_video_footage_with_LSTM/resolve/main/epoch4437val_loss0.6506.keras

## Reference

Samer Attrah. (2025). FER2013 blendshapes dataset example (Partial) [Data set]. Kaggle. https://doi.org/10.34740/KAGGLE/DSV/10716347

## Acknowledgement

Thanks to professor Marijn Jongerden, Jeroen Veen and Dixon Devasia for their help and guidance along the way, and thanks to Victor Hogeweij for his contribution to the Gaze project. Also, thanks Bhupinder Kaur, An Le, and Muhammad Reza for their insight and support at the early stages of the work.

## Citation
If you use this project in your research, please cite it as follows:
Attrah S. Emotion estimation from video footage with LSTM. arXiv preprint arXiv:2501.13432. 2025 Jan 23.

```bibtex
@article{attrah2025emotion,
  title={Emotion estimation from video footage with LSTM},
  author={Attrah, Samer},
  journal={arXiv preprint arXiv:2501.13432},
  year={2025}
}
