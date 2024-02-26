import numpy as np 
from alexnet import alexnet
from sklearn.model_selection import train_test_split
import tensorflow as tf

WIDTH = 480
HEIGHT = 360
LR = 0.001
EPOCHS = 13
OUTPUT = 9

MODEL_NAME = f"GTA_V_{LR}_alexnet_{EPOCHS}-epochs.model"

train_data = np.load('training_data_v1.npy', allow_pickle=True)

# Split the array into X and y
X = train_data[:, 0]
# X = np.array([x.reshape((WIDTH, HEIGHT, 1)) for x in X])
X = np.array([i[0] for i in train_data]).reshape(-1,WIDTH,HEIGHT,1)
Y = np.array([i[1] for i in train_data]).reshape(-1, OUTPUT)  # Select all rows from the second column (labels)

model = alexnet(WIDTH, HEIGHT, OUTPUT)

# Split the data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

X_train = tf.convert_to_tensor(X_train, dtype=tf.float32)
Y_train = tf.convert_to_tensor(Y_train, dtype=tf.float32)
# print(X_train.shape)
# print(Y_train.shape)

X_test = tf.convert_to_tensor(X_test, dtype=tf.float32)
Y_test = tf.convert_to_tensor(Y_test, dtype=tf.float32)

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LR),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.fit(X_train,Y_train,epochs=EPOCHS,validation_split=0.1, shuffle=True,
                    verbose=1, batch_size=32)

model.summary()

model.save(MODEL_NAME)