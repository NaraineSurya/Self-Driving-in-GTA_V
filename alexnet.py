import tensorflow as tf

def alexnet(width, height, lr):
    # Define input placeholder
    input_data = tf.placeholder(tf.float32, shape=[None, width, height, 1], name='input')

    # Convolutional layer 1
    conv1 = tf.layers.conv2d(input_data, filters=96, kernel_size=11, strides=4, activation=tf.nn.relu)

    # Max pooling layer 1
    pool1 = tf.layers.max_pooling2d(conv1, pool_size=3, strides=2)

    # Local response normalization layer 1
    norm1 = tf.nn.local_response_normalization(pool1)

    # Convolutional layer 2
    conv2 = tf.layers.conv2d(norm1, filters=256, kernel_size=5, activation=tf.nn.relu)

    # Max pooling layer 2
    pool2 = tf.layers.max_pooling2d(conv2, pool_size=3, strides=2)

    # Local response normalization layer 2
    norm2 = tf.nn.local_response_normalization(pool2)

    # Convolutional layer 3
    conv3 = tf.layers.conv2d(norm2, filters=384, kernel_size=3, activation=tf.nn.relu)

    # Convolutional layer 4
    conv4 = tf.layers.conv2d(conv3, filters=384, kernel_size=3, activation=tf.nn.relu)

    # Convolutional layer 5
    conv5 = tf.layers.conv2d(conv4, filters=256, kernel_size=3, activation=tf.nn.relu)

    # Max pooling layer 3
    pool3 = tf.layers.max_pooling2d(conv5, pool_size=3, strides=2)

    # Local response normalization layer 3
    norm3 = tf.nn.local_response_normalization(pool3)

    # Flatten layer
    flatten = tf.layers.flatten(norm3)

    # Fully connected layer 1
    fc1 = tf.layers.dense(flatten, units=4096, activation=tf.nn.tanh)

    # Dropout layer 1
    dropout1 = tf.layers.dropout(fc1, rate=0.5)

    # Fully connected layer 2
    fc2 = tf.layers.dense(dropout1, units=4096, activation=tf.nn.tanh)

    # Dropout layer 2
    dropout2 = tf.layers.dropout(fc2, rate=0.5)

    # Output layer
    output = tf.layers.dense(dropout2, units=17, activation=tf.nn.softmax)

    return output


# Sequential layer for alexnet

# def alexnet(width, height, lr):
#     model = tf.keras.Sequential([
#         tf.keras.layers.Conv2D(filters=96, kernel_size=11, strides=4, activation='relu', input_shape=(width, height, 1)),
#         tf.keras.layers.MaxPooling2D(pool_size=3, strides=2),
#         tf.keras.layers.Lambda(lambda x: tf.nn.local_response_normalization(x)),
#         tf.keras.layers.Conv2D(filters=256, kernel_size=5, activation='relu'),
#         tf.keras.layers.MaxPooling2D(pool_size=3, strides=2),
#         tf.keras.layers.Lambda(lambda x: tf.nn.local_response_normalization(x)),
#         tf.keras.layers.Conv2D(filters=384, kernel_size=3, activation='relu'),
#         tf.keras.layers.Conv2D(filters=384, kernel_size=3, activation='relu'),
#         tf.keras.layers.Conv2D(filters=256, kernel_size=3, activation='relu'),
#         tf.keras.layers.MaxPooling2D(pool_size=3, strides=2),
#         tf.keras.layers.Lambda(lambda x: tf.nn.local_response_normalization(x)),
#         tf.keras.layers.Flatten(),
#         tf.keras.layers.Dense(4096, activation='tanh'),
#         tf.keras.layers.Dropout(0.5),
#         tf.keras.layers.Dense(4096, activation='tanh'),
#         tf.keras.layers.Dropout(0.5),
#         tf.keras.layers.Dense(17, activation='softmax')
#     ])

#     return model

