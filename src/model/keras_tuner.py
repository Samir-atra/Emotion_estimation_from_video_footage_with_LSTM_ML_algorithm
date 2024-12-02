"""keras tuner for LSTM model experimenting on five hyperparameters """

import tensorflow as tf
import keras_tuner


def build_model(hp):
  """build a compiled keras LSTM model and return it"""
  
  learning_rate = hp.Float("lr", min_value=1e-6, max_value=1e-3, sampling="log")       # learning rate search range
  layer_u = hp.Int("lu", min_value=10, max_value=32, step=4)                           # layer units search range
  kernel_r = hp.Float("kr", min_value=1e-10, max_value=1e-5, sampling="log")           # kernel regularization search range
  acti_f = hp.Choice("af", ['selu', 'tanh', 'relu', 'leaky_relu'])                     # activation function search range
  weight_d = hp.Float("wd", min_value=1e-10, max_value=0.0009, sampling="log")         # weight decay search range
  
# model structure
  model = tf.keras.Sequential([
      tf.keras.layers.LSTM(units = 34, activation = 'selu', return_sequences= True, kernel_regularizer=tf.keras.regularizers.L2(l2=0.00000195)),
      tf.keras.layers.LSTM(units = 26, activation = acti_f, return_sequences= True, kernel_regularizer=tf.keras.regularizers.L2(l2=kernel_r)),
      tf.keras.layers.LSTM(units = layer_u, activation = acti_f, return_sequences= True, kernel_regularizer=tf.keras.regularizers.L2(l2=kernel_r)),
      tf.keras.layers.LSTM(units = layer_u, activation = acti_f, return_sequences= True, kernel_regularizer=tf.keras.regularizers.L2(l2=kernel_r)),
      tf.keras.layers.LSTM(units = layer_u, activation = acti_f, return_sequences= True, kernel_regularizer=tf.keras.regularizers.L2(l2=kernel_r)),
      tf.keras.layers.LSTM(units = layer_u, activation = acti_f, return_sequences= True, kernel_regularizer=tf.keras.regularizers.L2(l2=kernel_r)),
      tf.keras.layers.LSTM(units = 30, activation = acti_f, return_sequences= False, kernel_regularizer=tf.keras.regularizers.L2(l2=0.00000195)),
      tf.keras.layers.Dense(units = 3, activation = 'softmax'),
  ])
  
# Compiling the model
  model.compile(loss=tf.keras.losses.CategoricalCrossentropy(),
              optimizer= tf.keras.optimizers.Adam(learning_rate = learning_rate, global_clipnorm=1, amsgrad = True, weight_decay=weight_d),
              metrics = [tf.keras.metrics.CategoricalCrossentropy(), tf.keras.metrics.CategoricalAccuracy(), tf.keras.metrics.F1Score()])

  return model 


def experimenting(X_train, Y_train, X_val, y_val):
    """run keras tuner experiments for the model built using build_model 
    using random search"""
    
    build_model(keras_tuner.HyperParameters())

    tuner = keras_tuner.RandomSearch(                               # tuner configurations
        hypermodel=build_model,
        max_trials=15,
        objective=keras_tuner.Objective('val_loss', 'min'),
        executions_per_trial=1,
        overwrite=True,
        directory="/home/samer/Desktop/Big data Small Data/BDSD/Minor_project/emotion_estimation/",
        project_name="Emotion_estimation_tuning",
    )

    tuner.search_space_summary()

    tuner.search(x=X_train, y=Y_train, validation_data = (X_val,y_val), epochs=100, batch_size = 150)

    tuner.results_summary()
