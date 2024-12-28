"""LSTM model fitting using a manually built model 
with compiling and checkpoints"""


import keras



def train_model(x_train, y_train, x_val, y_val):
    """train the model for manually set hyperparameters"""
    # LSTM model final structure experimented with
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
                return_sequences=True,
                kernel_regularizer=keras.regularizers.L2(l2=0.00000195),
            ),
            keras.layers.Dense(units=3, activation="softmax"),
        ]
    )

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
            keras.metrics.F1Score(),
        ],
    )

    # save model checkpoints
    checkpoint_filepath = "ckpt/epoch:{epoch:02d}-val_loss:{val_loss:.4f}.keras"
    model_checkpoint_callback = (
        keras.callbacks.ModelCheckpoint(  # model checkpoint callback
            filepath=checkpoint_filepath,
            monitor="val_loss",
            mode="min",
            save_best_only=True,
            verbose=0,
        )
    )

    # earlystopping callback
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
        epochs=2000,
        batch_size=150,
        verbose=2,
        class_weight=class_weight,
        callbacks=[model_checkpoint_callback, early_stop],
    )
    keras.utils.plot_model(model, show_shapes=True, rankdir="LR")

    return model
