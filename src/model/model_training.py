"""
LSTM model fitting using a manually built model 
with compiling and checkpoints
"""


import keras
import tensorflow as tf
from dotenv import load_dotenv
import os
import sys

# Add the 'src' directory to the Python path to resolve module imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Load environment variables from .env file
load_dotenv()

from data.data_loading import load_training_data, load_validation
import datetime

log_dir = "logs/profile/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
# Profile batches 10 to 15 (inclusive)
# This collects performance data, including operation execution times (latency)
tboard_callback = tf.keras.callbacks.TensorBoard(
    log_dir=log_dir,
    profile_batch='10, 15'
)

def train_model(x_train, y_train, x_val, y_val):
    """train the model for manually set hyperparameters"""
    # LSTM model final structure experimented with and found using the keras tuner
    model = keras.Sequential(
        [
            keras.layers.LSTM(
                units=53,
                activation="selu",
                return_sequences=True,
                kernel_regularizer=keras.regularizers.L2(l2=0.00000195),
            ),
            keras.layers.LSTM(
                units=16,
                activation="selu",
                return_sequences=True,
                kernel_regularizer=keras.regularizers.L2(l2=0.00000195),
            ),
            keras.layers.LSTM(
                units=48,
                activation="selu",
                return_sequences=True,
                kernel_regularizer=keras.regularizers.L2(l2=0.00000195),
            ),
            keras.layers.LSTM(
                units=44,
                activation="selu",
                return_sequences=False,
                kernel_regularizer=keras.regularizers.L2(l2=0.00000195),
            ),
            keras.layers.Dense(units=3, activation="softmax"),
        ]
    )
    # compile the model
    model.compile(
        loss=keras.losses.CategoricalCrossentropy(),
        optimizer=keras.optimizers.AdamW(
            learning_rate = 1.0955e-06,
            global_clipnorm = 1,
            amsgrad = True,
            weight_decay = 5.468661421085422e-05,
        ),
        metrics=[
            keras.metrics.CategoricalCrossentropy(),
            keras.metrics.CategoricalAccuracy(),
            keras.metrics.F1Score(average="macro"),
        ],
    )

    # save model checkpoints
    checkpoint_filepath = "ckpt/epoch:{epoch:02d}-val_loss:{val_loss:.4f}.keras"
    model_checkpoint_callback = (
        keras.callbacks.ModelCheckpoint(  # model checkpoint callback
            filepath=checkpoint_filepath,
            monitor="val_loss",      # the check point takes the validation loss as main metric for preference
            mode="min",          # the best model checkpoint is the one with minimum validation loss
            save_best_only=True,
            verbose=0,
        )
    )

    # earlystopping callback stops the training after NOT having any 
    # improvment of 0.0001 in the categorical cross entropy for 500 epochs 
    early_stop = keras.callbacks.EarlyStopping(
        monitor="categorical_crossentropy",
        mode="min",
        min_delta=0.0001,
        patience=500,
        restore_best_weights=True,
        verbose=0,
    )

    # assigning weights for the classes (untuned)
    weight_for_0 = 1
    weight_for_1 = 1
    weight_for_2 = 1

    class_weight = {0: weight_for_0, 1: weight_for_1, 2: weight_for_2}

    # # Fitting the RNN to the Training set
    model.fit(
        x=x_train,
        y=y_train,
        validation_data=(x_val, y_val),
        epochs=20,
        batch_size=150,
        verbose=2,
        class_weight=class_weight,
        callbacks=[model_checkpoint_callback, early_stop, tboard_callback],   # use the callbaacks model checkpoint and early stop
    )
    keras.utils.plot_model(model, show_shapes=True, rankdir="LR")     # plot the model structure

    return model

if __name__ == "__main__":
    # Load training and validation data
    x_train, y_train = load_training_data()
    x_val, y_val = load_validation()
    train_model(x_train, y_train, x_val, y_val)
