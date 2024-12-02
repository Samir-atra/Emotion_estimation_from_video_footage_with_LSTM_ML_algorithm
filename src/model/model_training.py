"""LSTM model fitting using a manually built model 
with compiling and checkpoints"""

import numpy as np
import matplotlib.pyplot as plt
import cv2
import pathlib
import csv
import pandas as pd
import tensorflow as tf
import keras_tuner
import math
import time
import os


def train_model(X_train, Y_train, X_val, y_val):
    """train the model for manually set hyperparameters"""
    # LSTM model final structure experimented with
    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(units = 34, activation = 'selu', return_sequences= True, kernel_regularizer=tf.keras.regularizers.L2(l2=2.256393134751847e-06)),
        tf.keras.layers.LSTM(units = 26, activation = 'selu', return_sequences= True, kernel_regularizer=tf.keras.regularizers.L2(l2=2.256393134751847e-06)),
        tf.keras.layers.LSTM(units = 30, activation = 'selu', return_sequences= True, kernel_regularizer=tf.keras.regularizers.L2(l2=2.256393134751847e-06)),
        tf.keras.layers.LSTM(units = 30, activation = 'selu', return_sequences= True, kernel_regularizer=tf.keras.regularizers.L2(l2=2.256393134751847e-06)),
        tf.keras.layers.LSTM(units = 30, activation = 'selu', return_sequences= True, kernel_regularizer=tf.keras.regularizers.L2(l2=2.256393134751847e-06)),
        tf.keras.layers.LSTM(units = 30, activation = 'selu', return_sequences= True, kernel_regularizer=tf.keras.regularizers.L2(l2=2.256393134751847e-06)),
        tf.keras.layers.LSTM(units = 30, activation = 'selu', return_sequences= False, kernel_regularizer=tf.keras.regularizers.L2(l2=2.256393134751847e-06)),
        tf.keras.layers.Dense(units = 3, activation = 'softmax'),
    ])


    model.compile(loss=tf.keras.losses.CategoricalCrossentropy(),
                optimizer= tf.keras.optimizers.AdamW(learning_rate = 5.6761842602901e-05, global_clipnorm=1, amsgrad = True, weight_decay=5.468661421085422e-05),
                metrics = [tf.keras.metrics.CategoricalCrossentropy(),tf.keras.metrics.CategoricalAccuracy(), tf.keras.metrics.F1Score()])


    # save model checkpoints
    checkpoint_filepath = 'ckpt/epoch:{epoch:02d}-val_loss:{val_loss:.4f}.keras'
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(                  # model checkpoint callback
        filepath=checkpoint_filepath,
        monitor='val_loss',
        mode='min',
        save_best_only=True,
        verbose = 0)


    # earlystopping callback
    EStop = tf.keras.callbacks.EarlyStopping(monitor="categorical_crossentropy", mode = 'min', min_delta= 0.0001, patience=500, restore_best_weights=True, verbose=0)


    # assigning weights for the classes (untuned)
    weight_for_0 = 1
    weight_for_1 = 1
    weight_for_2 = 1


    class_weight = {0: weight_for_0, 1: weight_for_1, 2: weight_for_2}


    # # Fitting the RNN to the Training set
    model.fit(x=X_train, y=Y_train, validation_data = (X_val,y_val) ,epochs = 4000, batch_size=150, verbose=2, class_weight=class_weight, callbacks=[model_checkpoint_callback, EStop])
    tf.keras.utils.plot_model(model, show_shapes=True, rankdir="LR")
    
    return model
