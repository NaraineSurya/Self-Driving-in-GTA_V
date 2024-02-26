import numpy as np 
from collections import Counter
from random import shuffle
import cv2

train_data = np.load('training_data_v1.npy', allow_pickle=True)

for data in train_data:
    img = data[0]
    choice = data[1]
    cv2.imshow('test',img)
    print(choice)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break