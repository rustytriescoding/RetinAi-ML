import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import matplotlib.pyplot as plt 
import pandas as pd
import numpy as np
import os
from tqdm import tqdm

from RetinaDiseaseDataset import RetinaDiseaseDataset
from RetinaDiseaseClassifier import RetinaDiseaseClassifier

train_csv='../data/csvs/train.csv'
val_csv='../data/csvs/val.csv'
test_csv='../data/csvs/test.csv'

image_path='../data/ocular-disease-recognition-odir5k/ODIR-5K/Training Images'

IMAGE_SIZE = 128
data_transform = transforms.Compose([transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)), transforms.ToTensor()])

diagnosis_list = ['Normal', 'Diabetes', 'Glaucoma', 'Cataract', 'Age', 'Hypertension', 'Pathological Myopia', 'Other']

train_dataset = RetinaDiseaseDataset(
    train_csv,
    image_path,
    data_transform
)
    
val_dataset = RetinaDiseaseDataset(
    val_csv,
    image_path,
    data_transform
)

# Unused
test_dataset = RetinaDiseaseDataset(
    test_csv,
    image_path,
    data_transform
)

# Create DataLoaders for each subset
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False) # Unused

model = RetinaDiseaseClassifier(num_classes=8)

# load model
model.load_state_dict(torch.load('../models/retinai_resnet50_0.0.1.pth'))

# Simple training loop
num_epochs = 1
train_losses, val_losses = [], []

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(num_epochs):
    # Training
    model.train()
    running_loss = 0.0
    for images, labels in tqdm(train_loader, desc='Training loop'):
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * labels.size(0)
    train_loss = running_loss / len(train_loader.dataset)
    train_losses.append(train_loss)
    
    # Validation
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc='Validation loop'):
            images, labels = images.to(device), labels.to(device)
         
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * labels.size(0)
    val_loss = running_loss / len(val_loader.dataset)
    val_losses.append(val_loss)
    print(f"Epoch {epoch+1}/{num_epochs} - Train loss: {train_loss}, Validation loss: {val_loss}")

model_path = '../models/'
model_path = os.path.join(model_path, 'retinai_resnet50_0.0.1.pth')
torch.save(model.state_dict(), model_path)