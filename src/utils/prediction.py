"""pridict the class of a single image"""

import numpy as np
import tensorflow as tf

# new_model = tf.keras.models.load_model('LSTM_model_73%_test_acc')
predictions = []
for inst in X_test:
    inst = np.array(inst, dtype=np.float64)
    inst = np.reshape(inst, (1,52,1))

    y_pred = model.predict(inst, verbose=0)
    y_pred = np.argmax(y_pred)
    predictions.append(y_pred)
