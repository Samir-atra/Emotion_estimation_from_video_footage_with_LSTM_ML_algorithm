#!/bin/bash

# This script automates the setup of necessary environment variables and downloads
# MediaPipe assets required for the Emotion Estimation project.

# Exit immediately if a command exits with a non-zero status.
set -e

echo "Starting script for Emotion Estimation project setup..."

# =============================================================================
# 1. TensorFlow Environment Configuration
# -----------------------------------------------------------------------------
# Disable oneDNN optimizations for TensorFlow. This can sometimes resolve
# specific compatibility issues or errors, such as cuDNN factory registration
# errors, especially in mixed Keras/TensorFlow environments or specific GPU setups.
export TF_ENABLE_ONEDNN_OPTS=0
echo "TF_ENABLE_ONEDNN_OPTS set to 0 to potentially resolve TensorFlow/cuDNN issues."

# =============================================================================
# 2. Download MediaPipe Assets
# -----------------------------------------------------------------------------
# Download the MediaPipe Face Landmarker model. This model is essential for
# detecting facial landmarks and extracting blendshapes from images.
FACE_LANDMARKER_MODEL="face_landmarker_v2_with_blendshapes.task"
MODEL_URL="https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"

if [ ! -f "$FACE_LANDMARKER_MODEL" ]; then
    echo "Downloading MediaPipe Face Landmarker model..."
    wget -O "$FACE_LANDMARKER_MODEL" -q "$MODEL_URL"
    echo "Downloaded $FACE_LANDMARKER_MODEL"
else
    echo "$FACE_LANDMARKER_MODEL already exists. Skipping download."
fi

# Download a sample image for testing and visualizing blendshapes.
SAMPLE_IMAGE="image.png"
IMAGE_URL="https://storage.googleapis.com/mediapipe-assets/business-person.png"

if [ ! -f "$SAMPLE_IMAGE" ]; then
    echo "Downloading sample image..."
    wget -q -O "$SAMPLE_IMAGE" "$IMAGE_URL"
    echo "Downloaded $SAMPLE_IMAGE"
else
    echo "$SAMPLE_IMAGE already exists. Skipping download."
fi

# =============================================================================
# 3. Python Path Configuration
# -----------------------------------------------------------------------------
# Add the project's 'src' directory to the PYTHONPATH. This ensures that Python
# can find and import local modules (e.g., data, mediapipe_tools, model, utils)
# when scripts are run from the project root or other locations.
# Note: Adjust the path if your project structure differs or if running from a different location.
PROJECT_ROOT="$(dirname "$(readlink -f "$0")")"
SRC_DIR="$PROJECT_ROOT/src"
export PYTHONPATH="$PYTHONPATH:$SRC_DIR"
echo "Added $SRC_DIR to PYTHONPATH."
export PYTHONPATH=$PYTHONPATH:"/home/samer/Desktop/HAN stuff/Big data Small Data/BDSD/Minor_project/BDSD_Minor_Project/src"

# =============================================================================
# 4. Matplotlib Backend Configuration
# -----------------------------------------------------------------------------
# Set the Matplotlib backend to 'Agg'. This is a non-interactive backend that
# can be used to generate image files (e.g., PNGs) without requiring a GUI
# toolkit. This is useful for running scripts in environments without a display
# server (e.g., remote servers, CI/CD pipelines).
export MPLBACKEND=Agg
echo "Matplotlib backend set to Agg for non-interactive plotting."

echo "Script execution finished."