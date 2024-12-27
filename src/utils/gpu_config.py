"""
configures a logical GPU memory minimum limit and enable synchronous execution
"""

import tensorflow as tf

gpus = tf.config.list_physical_devices("GPU")
tf.config.experimental.set_synchronous_execution(True)

if gpus:
    try:
        tf.config.set_logical_device_configuration(
            gpus[0], [tf.config.LogicalDeviceConfiguration(memory_limit=3076)]
        )
        logical_gpus = tf.config.list_logical_devices("GPU")
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Virtual devices must be set before GPUs have been initialized
        print(e)
