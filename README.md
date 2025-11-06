# Emotion Estimation from Video Footage with LSTM

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![ResearchGate](https://img.shields.io/badge/ResearchGate-Paper-blue)](https://www.researchgate.net/publication/387438191_EMOTION_ESTIMATION_FROM_VIDEO_FOOTAGE_WITH_LSTM)

## Overview

This repository contains the code accompanying the research paper "Emotion estimation from video footage with LSTM" by Samer Attrah. The project focuses on developing and evaluating an LSTM-based model for emotion recognition from facial blendshapes extracted using MediaPipe. It includes scripts for data processing, augmentation, model training, hyperparameter tuning, and error analysis.

## Features

*   **MediaPipe Integration:** Utilizes MediaPipe for robust facial landmark and blendshape extraction.
*   **Dataset Cleaning & Balancing:** Scripts to filter out unreadable images and create class-balanced datasets.
*   **Image Augmentation:** Techniques to expand the training dataset and improve model generalization.
*   **LSTM Model:** Implementation of a Long Short-Term Memory (LSTM) neural network for sequential emotion data.
*   **Keras Tuner:** Hyperparameter optimization for the LSTM model using Keras Tuner.
*   **Model Evaluation:** Tools for evaluating model performance, including confusion matrix generation and error visualization.
*   **GPU Configuration:** Utility for configuring TensorFlow to optimize GPU memory usage.

## Installation

To set up the project locally, follow these steps:

### Prerequisites

*   Python 3.9 (recommended, as specified in `requirements.txt`)
*   Conda or Miniconda (for environment management)
*   Git

### 1. Clone the Repository

```bash
git clone https://github.com/SamerAttrah/Emotion_estimation_from_video_footage_with_LSTM.git
cd Emotion_estimation_from_video_footage_with_LSTM
```

### 2. Create and Activate Conda Environment

It is crucial to use the specified Python and Keras versions for compatibility.

```bash
conda create -n BlendFER_project python=3.9
conda activate BlendFER_project
pip install -r requirements.txt
```
**Note on Keras/TensorFlow Versions:** The project was developed with a mix of Keras 2.x and Keras 3.x functionalities. The `requirements.txt` file specifies `keras==2.15.0` and `tensorflow>=2.15.0` to ensure compatibility.

### 3. Download MediaPipe Assets

Download the necessary MediaPipe Face Landmarker model and a sample image for testing:

```bash
wget -O face_landmarker_v2_with_blendshapes.task -q https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task
wget -q -O image.png https://storage.googleapis.com/mediapipe-assets/business-person.png
```

### 4. Download Datasets

*   **FER2013 Dataset:** Download the original FER2013 dataset from [Kaggle](https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data). Place the `fer2013.csv` file in a suitable `Datasets` directory (e.g., `Datasets/fer2013.csv`).
*   **Pre-processed Blendshapes Dataset (Optional):** You can also download a pre-processed blendshapes dataset from [Kaggle](https://www.kaggle.com/datasets/samerattrah/fer2013-blendshapes-dataset-example-partial) to skip the initial data processing steps.

### 5. Download Pre-trained Model (Optional)

A pre-trained LSTM model is available for direct evaluation:
*   [HuggingFace](https://huggingface.co/SamerAttrah/Emotion_estimation_from_video_footage_with_LSTM/resolve/main/epoch4437val_loss0.6506.keras)
*   [Kaggle](https://www.kaggle.com/models/samerattrah/blendfer-lite)

Place the downloaded `.keras` model file in the project root or update the `SAVED_MODEL_PATH` in your `.env` file.

### 6. Environment Variables

Create a `.env` file in the project root directory and define the following paths:

```
FER2013_DATASET_PATH="/path/to/your/fer2013.csv"
TRAIN_DATASET="/path/to/your/training_set_full.csv"
VAL_DATASET="/path/to/your/validation_set_full.csv"
TEST_DATASET="/path/to/your/test_set_full.csv"
FULL_TEST_SET="/path/to/your/test_set_full_index.csv" # Used in error_analysis
FACE_LANDMARKER="face_landmarker_v2_with_blendshapes.task"
SAVED_MODEL_PATH="/path/to/your/epoch4437val_loss0.6506.keras"
KERAS_TUNER_EXPERIMENTS_DIR="keras_tuner_experiments"
```
**Note:** Update `/path/to/your/` with the actual absolute paths on your system.

## Usage

The project workflow typically involves data preparation, model training, and evaluation.

### Data Preparation

1.  **Clean and Balance Dataset:**
    Run `src/data/data_cleaning.py` to process the raw FER2013 dataset, filter images unreadable by MediaPipe, and create balanced training, validation, and test sets. This script will generate `training_set_full.csv`, `validation_set_full.csv`, and `test_set_full.csv`.
    ```bash
    python src/data/data_cleaning.py
    ```
2.  **Augment and Normalize Data:**
    Use `src/data/augmenting_and_normalizing.py` to augment the training data and normalize the datasets.
    ```bash
    python src/data/augmenting_and_normalizing.py
    ```
3.  **Create Blendshapes Dataset:**
    The `src/mediapipe_tools/blendshapes_dataset.py` script can be used to extract blendshapes from images and save them to a CSV.
    ```bash
    python src/mediapipe_tools/blendshapes_dataset.py
    ```
4.  **Index Test Data for Error Analysis:**
    The `src/data/dataset_indexing.py` script adds an index column to the test dataset, which is useful for tracking misclassified images.
    ```bash
    python src/data/dataset_indexing.py
    ```

### Model Training & Tuning

1.  **Train the LSTM Model:**
    Execute `src/model/model_training.py` to train the LSTM model with predefined hyperparameters.
    ```bash
    python src/model/model_training.py
    ```
2.  **Hyperparameter Tuning with Keras Tuner:**
    Run `src/model/keras_tuner_experimenter.py` to perform hyperparameter optimization using Keras Tuner's RandomSearch.
    ```bash
    python src/model/keras_tuner_experimenter.py
    ```

### Evaluation & Analysis

1.  **Evaluate Model Performance:**
    Use `src/model/evaluation.py` to load a trained model and evaluate its performance on the test set.
    ```bash
    python src/model/evaluation.py
    ```
2.  **Visualize Errors:**
    The `src/utils/error_analysis.py` script helps visualize misclassified images from the test set.
    ```bash
    python src/utils/error_analysis.py
    ```
3.  **Visualize Test Data Blendshapes:**
    `src/utils/test_data_visualization.py` can be used to visualize the distribution of blendshapes in the test dataset across different emotion classes.
    ```bash
    python src/utils/test_data_visualization.py
    ```
4.  **Generate Confusion Matrix:**
    The `src/utils/confusion_matrix.py` script can be adapted to generate and analyze the confusion matrix.
    ```bash
    # This script is typically called internally by evaluation or error analysis,
    # but can be run standalone if integrated with data loading.
    # python src/utils/confusion_matrix.py
    ```

## Project Structure

```
.
├── .git/
├── ckpt/                         # Model checkpoints
├── logs/                         # TensorBoard logs
├── src/
│   ├── data/                     # Scripts for data loading, cleaning, augmentation, and indexing
│   │   ├── augmenting_and_normalizing.py
│   │   ├── data_cleaning.py
│   │   ├── data_loading.py
│   │   ├── data_processing.py
│   │   └── dataset_indexing.py
│   ├── mediapipe_tools/          # Utilities for MediaPipe integration and blendshape processing
│   │   ├── blendshapes_dataset.py
│   │   ├── choosing_blendshapes.py
│   │   └── visualizing_and_setup.py
│   ├── model/                    # Model definition, training, evaluation, and hyperparameter tuning
│   │   ├── evaluation.py
│   │   ├── keras_tuner_experimenter.py
│   │   └── model_training.py
│   └── utils/                    # General utility scripts (e.g., CSV writer, error analysis, GPU config)
│       ├── confusion_matrix.py
│       ├── csv_writer.py
│       ├── error_analysis.py
│       ├── gpu_config.py
│       ├── prediction_and_latency.py
│       └── test_data_visualization.py
├── .gitignore
├── epoch4437val_loss0.6506.keras # Example pre-trained model
├── foto.png                      # Example model plot
├── LICENSE
├── model.png                     # Another example model plot
├── README.md
├── requirements.txt
└── script.sh                     # Utility script (e.g., for downloading assets)
```

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
```