import numpy as np 
from alexnet import alexnet
from sklearn.model_selection import train_test_split

WIDTH = 80
HEIGHT = 60
LR = 0.001
EPOCHS = 8

model = alexnet(WIDTH, HEIGHT, LR)

train_data = np.load('training_data_v2.npy')

# Split the array into X and y
X = train_data[:, 0]
X.reshape(-1, WIDTH, HEIGHT, 1)
Y = train_data[:, 1]  # Select all rows from the second column (labels)

# Split the data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

model.fit(X_train,Y_train)