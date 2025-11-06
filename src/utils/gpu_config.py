"""Module for configuring GPU settings for TensorFlow.

This script attempts to configure a logical GPU memory limit and enables
synchronous execution for TensorFlow. The GPU memory configuration is
currently commented out for debugging purposes related to cuDNN registration.
"""

import tensorflow as tf

# List available physical GPUs
gpus = tf.config.list_physical_devices("GPU")

# Enable synchronous execution for easier debugging and predictable behavior.
# This can be commented out for performance in production environments.
tf.config.experimental.set_synchronous_execution(True)

if gpus:
    try:
        # Configure logical device memory limit for the first GPU.
        # This line is commented out to avoid potential cuDNN registration issues
        # during initial setup or debugging.
        tf.config.set_logical_device_configuration(
            gpus[0], [tf.config.LogicalDeviceConfiguration(memory_limit=3076)]    # Adjust allocated GPU-RAM
        )
        logical_gpus = tf.config.list_logical_devices("GPU")
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Catch and print runtime errors, typically if virtual devices are set after GPU initialization
        print(e)
