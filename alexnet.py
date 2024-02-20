import tensorflow as tf

"""
Implementing AlexNet CNN Architecture Using TensorFlow 2.0+ and Keras
https://towardsdatascience.com/implementing-alexnet-cnn-architecture-using-tensorflow-2-0-and-keras-2113e090ad98
"""

# Sequential layer for alexnet

def alexnet(width, height, lr):
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(filters=96, kernel_size=11, strides=4, activation='relu', input_shape=(width, height, 1)),
        tf.keras.layers.MaxPooling2D(pool_size=3, strides=2),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(filters=256, kernel_size=5, activation='relu'),
        tf.keras.layers.MaxPooling2D(pool_size=3, strides=2),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(filters=384, kernel_size=3, activation='relu'),
        tf.keras.layers.Conv2D(filters=384, kernel_size=3, activation='relu'),
        tf.keras.layers.Conv2D(filters=256, kernel_size=3, activation='relu'),
        tf.keras.layers.MaxPooling2D(pool_size=3, strides=2),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(4096, activation='tanh'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(4096, activation='tanh'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(17, activation='softmax')
    ])

    return model

