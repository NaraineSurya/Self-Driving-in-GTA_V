import numpy as np
from alexnet import AlexNet
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from random import shuffle

WIDTH = 480
HEIGHT = 360
LR = 0.001
EPOCHS = 9
OUTPUT = 9
MODEL_NAME = f"GTA_V_{LR}_alexnet_{EPOCHS}_epochs.pth"

model = AlexNet(num_classes=OUTPUT)

no_data = 3

TOTAL_DATA = []

for i in range(1, no_data + 1):
    train_data = np.load('d:/surya/Dataset/training_data_v{}.npy'.format(i), allow_pickle=True)
    shuffle(train_data)
    TOTAL_DATA.extend(train_data)

X = np.array([i[0] for i in TOTAL_DATA]).reshape(-1, 3, WIDTH, HEIGHT)
Y = np.array([i[1] for i in TOTAL_DATA])

print(X.shape)
print(Y.shape)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Define a data generator function
def data_generator(X, Y, batch_size):
    num_samples = len(X)
    indices = list(range(num_samples))
    while True:
        shuffle(indices)
        for i in range(0, num_samples, batch_size):
            batch_indices = indices[i:i+batch_size]
            batch_X = X[batch_indices]
            batch_Y = Y[batch_indices]
            yield torch.tensor(batch_X, dtype=torch.float32), torch.tensor(batch_Y, dtype=torch.long)

# Create DataLoader using data generator for training data
batch_size = 10
train_generator = data_generator(X_train, Y_train, batch_size)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(torch.cuda.is_available())

# Move model to GPU
model.to(device)

print(X.shape)
print(Y.shape)

# Training loop
num_batches = len(X_train) // batch_size
for epoch in range(EPOCHS):
    running_loss = 0.0
    for _ in range(num_batches):
        inputs, labels = next(train_generator)
        inputs, labels = inputs.to(device), labels.to(device)  # Move tensors to GPU
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {running_loss / num_batches}")

# Save the trained model
torch.save(model.state_dict(), MODEL_NAME)
