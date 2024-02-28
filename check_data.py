import numpy as np 
from collections import Counter
from random import shuffle
import cv2

train_data = np.load('Dataset/training_data_v1.npy', allow_pickle=True)

WIDTH = 480
HEIGHT = 360

for data in train_data:
    img = data[0]
    choice = data[1]
    # screen = cv2.resize(screen, (WIDTH,HEIGHT))
    cv2.imshow('test',img)

    print(choice)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break