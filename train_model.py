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
    train_data = np.load('Dataset/training_data_v{}.npy'.format(i), allow_pickle=True)
    shuffle(train_data)
    TOTAL_DATA.extend(train_data)

X = np.array([i[0] for i in TOTAL_DATA]).reshape(-1, 3, WIDTH, HEIGHT)
Y = np.array([i[1] for i in TOTAL_DATA])

print(X.shape)
print(Y.shape)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Convert numpy arrays to PyTorch tensors
X_train = torch.from_numpy(X_train).float()
Y_train = torch.from_numpy(Y_train).long()
X_test = torch.from_numpy(X_test).float()
Y_test = torch.from_numpy(Y_test).long()

# Create DataLoader
train_dataset = TensorDataset(X_train, Y_train)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

# Training loop
for epoch in range(EPOCHS):
    running_loss = 0.0
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {running_loss / len(train_loader)}")

# Save the trained model
torch.save(model.state_dict(), MODEL_NAME)
