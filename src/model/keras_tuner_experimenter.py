"""Module for hyperparameter tuning of an LSTM model using Keras Tuner.

This module defines an LSTM model architecture and uses Keras Tuner's
RandomSearch algorithm to experiment with various hyperparameters such as
learning rate, layer units, kernel regularization, activation functions, and weight decay.
"""

import keras
import keras_tuner
from dotenv import load_dotenv
import os
import sys

# Add the 'src' directory to the Python path to resolve module imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Load environment variables from .env file
load_dotenv()

from data.data_loading import load_training_data, load_validation

def build_model(hp):
    """
    Builds a compiled Keras LSTM model with hyperparameters to be experimented on.

    This function defines the architecture of the LSTM model, including multiple
    LSTM layers and a final dense layer with softmax activation. It incorporates
    hyperparameter search spaces for key model parameters like learning rate,
    number of LSTM units, kernel regularization, and activation functions.

    Args:
        hp (keras_tuner.HyperParameters): An instance of Keras Tuner's HyperParameters class,
                                          used to define the search space for hyperparameters.

    Returns:
        keras.Model: The compiled Keras LSTM model with hyperparameters set by Keras Tuner.
    """

    # Define hyperparameter search spaces for tuning
    learning_rate = hp.Float("lr", min_value=1e-6, max_value=1e-3, sampling="log")        # Learning rate search space
    layer_u = hp.Int("lu", min_value=10, max_value=32, step=4)                          # Number of units in LSTM layers search space
    kernel_r = hp.Float("kr", min_value=1e-10, max_value=1e-5, sampling="log")            # Kernel regularization (L2) search space limits
    acti_f = hp.Choice("af", ["selu", "tanh", "relu", "leaky_relu"])                     # Activation function choices search space
    weight_d = hp.Float("wd", min_value=1e-10, max_value=0.0009, sampling="log")          # Weight decay search space limits


    # Define the model structure using Keras Sequential API
    model = keras.Sequential(
        [
            keras.layers.LSTM(
                units=34,
                activation="selu", # Fixed activation for the first layer
                return_sequences=True,
                kernel_regularizer=keras.regularizers.L2(l2=0.00000195),
            ),
            keras.layers.LSTM(
                units=26,
                activation=acti_f, # Tunable activation function
                return_sequences=True,
                kernel_regularizer=keras.regularizers.L2(l2=kernel_r), # Tunable kernel regularization
            ),
            keras.layers.LSTM(
                units=layer_u, # Tunable number of units
                activation=acti_f,
                return_sequences=True,
                kernel_regularizer=keras.regularizers.L2(l2=kernel_r),
            ),
            keras.layers.LSTM(
                units=layer_u,
                activation=acti_f,
                return_sequences=True,
                kernel_regularizer=keras.regularizers.L2(l2=kernel_r),
            ),
            keras.layers.LSTM(
                units=layer_u,
                activation=acti_f,
                return_sequences=True,
                kernel_regularizer=keras.regularizers.L2(l2=kernel_r),
            ),
            keras.layers.LSTM(
                units=layer_u,
                activation=acti_f,
                return_sequences=True,
                kernel_regularizer=keras.regularizers.L2(l2=kernel_r),
            ),
            keras.layers.LSTM(
                units=30,
                activation=acti_f,
                return_sequences=False, # Last LSTM layer does not return sequences
                kernel_regularizer=keras.regularizers.L2(l2=0.00000195),
            ),
            keras.layers.Dense(units=3, activation="softmax"), # Output layer for 3 classes
        ]
    )

    # Compile the model with a tunable optimizer and metrics
    model.compile(
        loss=keras.losses.CategoricalCrossentropy(), # Categorical cross-entropy for one-hot encoded labels
        optimizer=keras.optimizers.Adam(
            learning_rate=learning_rate, # Tunable learning rate
            global_clipnorm=1, # Global norm clipping for gradients
            amsgrad=True,
            weight_decay=weight_d, # Tunable weight decay
        ),
        metrics=[
            keras.metrics.CategoricalCrossentropy(), # Monitor classification loss
            keras.metrics.CategoricalAccuracy(),     # Monitor classification accuracy
            keras.metrics.F1Score(),                 # Monitor F1-score
        ],
    )

    return model


def experimenting(x_train, y_train, x_val, y_val):
    """
    Runs Keras Tuner experiments for the LSTM model using the RandomSearch algorithm.

    This function initializes a `RandomSearch` tuner with the `build_model` function,
    configures the search objective (minimizing validation loss), and then executes
    the hyperparameter search across the defined search spaces. It prints summaries
    of the search space and the results.

    Args:
        x_train (np.array): The blendshapes (features) of the training set.
        y_train (np.array): The one-hot encoded labels of the training set.
        x_val (np.array): The blendshapes (features) of the validation set.
        y_val (np.array): The one-hot encoded labels of the validation set.

    """

    build_model(keras_tuner.HyperParameters()) # Instantiate a dummy model to build the search space

    # Initialize Keras Tuner's RandomSearch algorithm
    tuner = keras_tuner.RandomSearch(
        hypermodel=build_model,
        max_trials=15, # Maximum number of hyperparameter combinations to try
        objective=keras_tuner.Objective("val_loss", "min"),   # Objective is to minimize validation loss
        executions_per_trial=1, # Number of models to train for each trial (1 for efficiency)
        overwrite=True, # Overwrite previous results in the directory
        directory=os.getenv("KERAS_TUNER_EXPERIMENTS_DIR"), # Directory to save experiment logs and checkpoints
        project_name="Emotion_estimation_tuning", # Name of the Keras Tuner project
    )

    tuner.search_space_summary() # Print a summary of the hyperparameter search space

    # Run the hyperparameter search experiments
    tuner.search(
        x=x_train, y=y_train, validation_data=(x_val, y_val), epochs=100, batch_size=150
    )  

    tuner.results_summary() # Print a summary of the best performing trials


if __name__ == "__main__":
    # Load training and validation data
    x_train, y_train = load_training_data()
    x_val, y_val = load_validation()
    # Run the hyperparameter experimentation
    experimenting(x_train, y_train, x_val, y_val)