import numpy as np 
from collections import Counter
from random import shuffle
import cv2

train_data = np.load('Dataset/training_data_v30.npy', allow_pickle=True)
# train_data = np.load('d://surya/Dataset', allow_pickle=True)

WIDTH = 256
HEIGHT = 144

for data in train_data:
    img = data[0]
    choice = data[1]
    img_resized = cv2.resize(img, (WIDTH, HEIGHT))
    cv2.imshow('test', img_resized)

    print(choice)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break
