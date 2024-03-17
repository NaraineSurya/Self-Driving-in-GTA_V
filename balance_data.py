import numpy as np
import pandas as pd
from collections import Counter
from random import shuffle

no_data = 44
TOTAL_DATA = []

for i in range(1, no_data + 1):
    train_data = np.load('Dataset/training_data_v{}.npy'.format(i), allow_pickle=True)
    shuffle(train_data)
    TOTAL_DATA.extend(train_data)

print(len(TOTAL_DATA))

df = pd.DataFrame(TOTAL_DATA)
print(df.head())
print(Counter(df[1].apply(str)))

w = []
a = []
s = []
d = []
wa = []
wd = []
sa = []
sd = []
nk = []

shuffle(TOTAL_DATA)

for data in TOTAL_DATA:
    img = data[0]
    choice = data[1]

    if np.array_equal(choice, [1,0,0,0,0,0,0,0,0]):
        w.append([img,choice])
    elif np.array_equal(choice, [0,1,0,0,0,0,0,0,0]):
        s.append([img,choice])
    elif np.array_equal(choice, [0,0,1,0,0,0,0,0,0]):
        a.append([img,choice])
    elif np.array_equal(choice, [0,0,0,1,0,0,0,0,0]):
        d.append([img,choice])
    elif np.array_equal(choice, [0,0,0,0,1,0,0,0,0]):
        wa.append([img,choice])
    elif np.array_equal(choice, [0,0,0,0,0,1,0,0,0]):
        wd.append([img,choice])
    elif np.array_equal(choice, [0,0,0,0,0,0,1,0,0]):
        sa.append([img,choice])
    elif np.array_equal(choice, [0,0,0,0,0,0,0,1,0]):
        sd.append([img,choice])
    elif np.array_equal(choice, [0,0,0,0,0,0,0,0,1]):
        nk.append([img,choice])
    else :
        print("no matches")

lengths = {'w': len(w), 's': len(s), 'a': len(a), 'd': len(d), 'wa': len(wa), 'wd': len(wd), 'sa': len(sa), 'sd': len(sd), 'nk': len(nk)}

min_length_key = min(lengths, key=lengths.get)
print("Variable with minimum length:", min_length_key)

# Ensure all lists have the same length
min_length = min(len(w), len(s), len(a), len(d), len(wa), len(wd), len(sa), len(sd), len(nk))
w = w[:min_length]
s = s[:min_length]
a = a[:min_length]
d = d[:min_length]
wa = wa[:min_length]
wd = wd[:min_length]
sa = sa[:min_length]
sd = sd[:min_length]
nk = nk[:min_length]

final_data =  w + s + a + d + wa + wd + sa + sd + nk

shuffle(final_data)
print(len(final_data))
final_data_array = np.array(final_data, dtype=object)


batch_size = 10000  # Adjust the batch size as needed
num_batches = len(final_data_array) // batch_size + 1

for i in range(num_batches):
    start_idx = i * batch_size
    end_idx = min((i + 1) * batch_size, len(final_data_array))
    batch_data = final_data_array[start_idx:end_idx]
    np.save('balanced_data_batch{}.npy'.format(i), batch_data)
