"""pridict the class of a single image"""

import numpy as np
import keras


def generate_prediction(x_test):
    """use a saved model to generate predictions to the test set

    Args:
        x_test (list): the blendshapes of the test set

    Returns:
        predictions (list): the predicted classes of the test set
    """
    model = keras.models.load_model("LSTM_model_73%_test_acc")   # load the model to be used for prediction
    predictions = []
    for inst in x_test:
        inst = np.array(inst, dtype=np.float64)
        inst = np.reshape(inst, (1, 52, 1))     

        y_pred = model.predict(inst, verbose=0)   # predict the labels for the images
        y_pred = np.argmax(y_pred)
        predictions.append(y_pred)
    return predictions
