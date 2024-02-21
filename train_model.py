import numpy as np 
from alexnet import alexnet
from sklearn.model_selection import train_test_split
import tensorflow as tf

WIDTH = 80
HEIGHT = 60
LR = 0.001
EPOCHS = 8

MODEL_NAME = f"San-self-driving-car {LR} {'alexnet'} {EPOCHS}-epochs.model"

train_data = np.load('training_data_v2.npy', allow_pickle=True)

# Split the array into X and y
X = train_data[:, 0]
# X = np.array([x.reshape((WIDTH, HEIGHT, 1)) for x in X])
X = np.array([i[0] for i in train_data]).reshape(-1,WIDTH,HEIGHT,1)
Y = np.array([i[1] for i in train_data]).reshape(-1, 3)  # Select all rows from the second column (labels)

model = alexnet(WIDTH, HEIGHT, LR)

# Split the data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

X_train = tf.convert_to_tensor(X_train, dtype=tf.float32)
print(X_train.shape)
print(Y_train.shape)
Y_train = tf.convert_to_tensor(Y_train, dtype=tf.float32)

X_test = tf.convert_to_tensor(X_test, dtype=tf.float32)
Y_test = tf.convert_to_tensor(Y_test, dtype=tf.float32)

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LR),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.fit(X_train,Y_train,epochs=EPOCHS,validation_split=0.1, shuffle=True,
                    verbose=1, batch_size=64)

model.summary()

model.save(MODEL_NAME)

"""
Tensorboard for training, validation, and evaluation of the network...
root_logdir = os.path.join(os.curdir, "logs\\fit\\")
def get_run_logdir():
    run_id = time.strftime("run_%Y_%m_%d-%H_%M_%S")
    return os.path.join(root_logdir, run_id)
run_logdir = get_run_logdir()
tensorboard_cb = keras.callbacks.TensorBoard(run_logdir)

Open up a terminal at the directory level where the TensorBoard log folder exists and run the following command:
tensorboard --logdir logs
"""