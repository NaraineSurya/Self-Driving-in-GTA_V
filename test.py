import numpy as np 
from alexnet import alexnet
from sklearn.model_selection import train_test_split
# import tensorflow as tf
from random import shuffle

WIDTH = 320
HEIGHT = 240
LR = 0.001
EPOCHS = 9
OUTPUT = 9
MODEL_NAME = f"GTA_V_{LR}_alexnet_{EPOCHS}_epochs.model"

# model = alexnet(WIDTH, HEIGHT, LR)

no_data = 9


train_data = np.load('Dataset/training_data_v1.npy', allow_pickle=True)
X = np.array([i[0] for i in train_data]).reshape(-1,WIDTH,HEIGHT,3)
Y = np.array([i[1] for i in train_data]).reshape(-1, OUTPUT) 
print(X.shape)
print(Y.shape)
        