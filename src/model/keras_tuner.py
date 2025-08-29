"""keras tuner for LSTM model experimenting on five hyperparameters """

import keras
import keras_tuner


def build_model(hp):
    """
    build a compiled keras LSTM model with ranges of hyperparameters to be experimented on

    Args:
        hp (keras_tuner.HyperParameters): keras tuner function to run the experiments

    Returns:
        model (keras.model): the compiled LSTM model with keras_tuner hyperparameters
    """

    # hyper parameters ranges for tuning
    learning_rate = hp.Float("lr", min_value=1e-6, max_value=1e-3, sampling="log")    # set the learning rate search space
    layer_u = hp.Int("lu", min_value=10, max_value=32, step=4)                  # create the limits of the search space of the layer's units
    kernel_r = hp.Float("kr", min_value=1e-10, max_value=1e-5, sampling="log")    # set the kernel regularization limits search space limits
    acti_f = hp.Choice("af", ["selu", "tanh", "relu", "leaky_relu"])       # set the choices of the activation function search space
    weight_d = hp.Float("wd", min_value=1e-10, max_value=0.0009, sampling="log")    # set the weight decay search space limits


    # model structure
    model = keras.Sequential(
        [
            keras.layers.LSTM(
                units=34,
                activation="selu",
                return_sequences=True,
                kernel_regularizer=keras.regularizers.L2(l2=0.00000195),
            ),
            keras.layers.LSTM(
                units=26,
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
                units=layer_u,
                activation=acti_f,
                return_sequences=True,
                kernel_regularizer=keras.regularizers.L2(l2=kernel_r),
            ),
            keras.layers.LSTM(
                units=30,
                activation=acti_f,
                return_sequences=False,
                kernel_regularizer=keras.regularizers.L2(l2=0.00000195),
            ),
            keras.layers.Dense(units=3, activation="softmax"),
        ]
    )

    # Compiling the model
    model.compile(
        loss=keras.losses.CategoricalCrossentropy(),
        optimizer=keras.optimizers.Adam(
            learning_rate=learning_rate,
            global_clipnorm=1,
            amsgrad=True,
            weight_decay=weight_d,
        ),
        metrics=[
            keras.metrics.CategoricalCrossentropy(),
            keras.metrics.CategoricalAccuracy(),
            keras.metrics.F1Score(),
        ],
    )

    return model


def experimenting(x_train, y_train, x_val, y_val):
    """
    run keras tuner experiments for the model built using build_model
    using random search

    Args:
        x_train (np.array): the blendshapes of the training set
        y_train (np.array): the labels of the training set
        x_val (np.array): the blendshapes of the validation set
        y_val (np.array): the labels of the validation set

    """

    build_model(keras_tuner.HyperParameters())

    tuner = keras_tuner.RandomSearch(                  # using the random search algorithm to choose the values to the experiments
        hypermodel=build_model,
        max_trials=15,
        objective=keras_tuner.Objective("val_loss", "min"),   # conduct the experiments to minimize the validation loss
        executions_per_trial=1,
        overwrite=True,
        directory="/home/samer/Desktop/Big data Small Data/BDSD/Minor_project/emotion_estimation/",
        project_name="Emotion_estimation_tuning",
    )

    tuner.search_space_summary()

    tuner.search(
        x=x_train, y=y_train, validation_data=(x_val, y_val), epochs=100, batch_size=150
    )  # run the search experiments

    tuner.results_summary()
