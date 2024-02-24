import numpy as np
import pandas as pd
from collections import Counter
from random import shuffle
import cv2


train_data = np.load('training_data.npy', allow_pickle=True)
print(len(train_data))

df = pd.DataFrame(train_data)
print(df.head())
print(Counter(df[1].apply(str)))

lefts = []
rights = []
forwards = []

shuffle(train_data)

for data in train_data:
    img = data[0]
    choice = data[1]

    if np.array_equal(choice, [1, 0, 0]):
        lefts.append([img,choice])
    elif np.array_equal(choice, [0, 1, 0]):
        forwards.append([img,choice])
    elif np.array_equal(choice, [0, 0, 1]):
        rights.append([img,choice])
    else :
        print("no matches")

# Ensure all lists have the same length
min_length = min(len(lefts), len(forwards), len(rights))
lefts = lefts[:min_length]
forwards = forwards[:min_length]
rights = rights[:min_length]

final_data =  forwards + lefts + rights

shuffle(final_data)
print(len(final_data))
final_data_array = np.array(final_data, dtype=object)
np.save('training_data_v2.npy',final_data_array)


# for data in train_data:
#     img = data[0]
#     choice = data[1]
#     cv2.imshow('test',img)
#     print(choice)
#     if cv2.waitKey(25)  & 0xFF == ord('q'):
#         cv2.destroyAllWindows()
#         break
