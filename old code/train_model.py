import numpy as np 
from alexnet import alexnet
from sklearn.model_selection import train_test_split
import tensorflow as tf
from random import shuffle

WIDTH = 320
HEIGHT = 240
LR = 0.001
EPOCHS = 9
OUTPUT = 9
MODEL_NAME = f"GTA_V_{LR}_alexnet_{EPOCHS}_epochs.model"

model = alexnet(WIDTH, HEIGHT, LR)

no_data = 3

TOTAL_DATA = []

for i in range(1,no_data+1):
    train_data = np.load('Dataset/training_data_v{}.npy'.format(i), allow_pickle=True)
    shuffle(train_data)
    TOTAL_DATA.extend(train_data)

X = np.array([i[0] for i in train_data]).reshape(-1,WIDTH,HEIGHT,3)
Y = np.array([i[1] for i in train_data]).reshape(-1, OUTPUT)
print(X.shape)
print(Y.shape)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LR),
              loss='categorical_crossentropy',
              metrics=['accuracy'])
        

model.fit(X_train,Y_train,epochs=EPOCHS,validation_data=(X_test, Y_test), shuffle=True,
            verbose=1, batch_size=32)

model.save(MODEL_NAME)


'''
Bro make sure that loading data is done before the training loop. And also the model compiling part.
'''