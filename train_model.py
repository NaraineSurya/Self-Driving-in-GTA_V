import numpy as np
from alexnet import AlexNet
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from tqdm import tqdm
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from random import shuffle
from torch.cuda.amp import autocast

WIDTH = 480
HEIGHT = 360
LR = 3e-4
EPOCHS = 2
OUTPUT = 9
MODEL_NAME = f"GTA_V_{LR}_alexnet_{EPOCHS}_epochs.pth"

model = AlexNet(num_classes=OUTPUT).to('cuda')

no_data = 2

TOTAL_DATA = []

for i in range(1, no_data + 1):
    train_data = np.load('d:/surya/Dataset/training_data_v{}.npy'.format(i), allow_pickle=True)
    shuffle(train_data)
    TOTAL_DATA.extend(train_data)


class data(Dataset):
    def __init__(self):
        super().__init__()
    
    def __len__(self):
        return len(TOTAL_DATA)
    
    def __getitem__(self, index):
        return torch.tensor(TOTAL_DATA[index][0], dtype=torch.float32).reshape(3, HEIGHT, WIDTH), torch.tensor(TOTAL_DATA[index][1], dtype=torch.float32).reshape(OUTPUT)


# Create DataLoader using data generator for training data
train_data = data()
loader = DataLoader(train_data, batch_size=12)

print('data loaded')

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

print('starting training')

# Define scaler for automatic mixed precision
scaler = torch.cuda.amp.GradScaler()

for epoch in range(EPOCHS):
    for i, (x, y) in enumerate(tqdm(loader)):
        x = x.reshape(-1, 3, HEIGHT, WIDTH).cuda()
        y = y.reshape(-1, OUTPUT).cuda()
        
        optimizer.zero_grad()
        
        # Automatic mixed precision training
        with autocast():
            outputs = model(x)
            loss = criterion(outputs, y)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        if i % 20 == 0 and i != 0:
            print(f'Epoch: {epoch}, Loss: {loss.item()}')

# Save the trained model
torch.save(model.state_dict(), MODEL_NAME)
