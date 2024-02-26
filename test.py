import numpy as np 
from alexnet import alexnet
from sklearn.model_selection import train_test_split
import tensorflow as tf

WIDTH = 480
HEIGHT = 360
LR = 0.001
EPOCHS = 13

MODEL_NAME = f"GTA_V_{LR}_alexnet_{EPOCHS}-epochs.model"

train_data = np.load('training_data_v2.npy', allow_pickle=True)

# Split the array into X and y
X = train_data[:, 0]
print(X[0].shape)
X = np.array([x.reshape((WIDTH, HEIGHT, 1)) for x in X])
print(X.shape)
Y = train_data[:, 1]
print(Y[0].shape)
