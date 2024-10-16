import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import timm

import matplotlib.pyplot as plt # For data viz
import pandas as pd
import numpy as np
import sys
import os
from tqdm import tqdm

from PIL import Image

csv_path='../data/ocular-disease-recognition-odir5k/full_df.csv'
csv = pd.read_csv(csv_path)

train_path='../data/ocular-disease-recognition-odir5k/ODIR-5K/Training Images'
test_path='../data/ocular-disease-recognition-odir5k/ODIR-5K/Testing Images'

IMAGE_SIZE = 128
data_transform = transforms.Compose([transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)), transforms.ToTensor()])

diagnosis_list = ['Normal', 'Diabetes', 'Glaucoma', 'Cataract', 'Age', 'Hypertension', 'Pathological Myopia', 'Other']

class OcularDiseaseDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.df = pd.read_csv(csv_file) 
        self.transform = transform 
        self.root_dir = root_dir
        
        # Define a mapping from string labels to integers
        self.label_mapping = {
            "N": 0,  # Normal
            "D": 1,  # Diabetes
            "G": 2,  # Glaucoma
            "C": 3,  # Cataract
            "A": 4,  # Age-related Macular Degeneration
            "H": 5,  # Hypertension
            "M": 6,  # Pathological Myopia
            "O": 7   # Other diseases/abnormalities
        }

    def __len__(self):
        return len(self.df)  

    def __getitem__(self, index):
        image_path = os.path.join(self.root_dir,
                                self.df.filename[index])
        image = Image.open(image_path)
        
        label = self.df.labels[index]
        label = label.strip("[]").strip("'\"")  # Remove brackets from cell value
        
        # Convert the label to int
        label = self.label_mapping.get(label, -1)  # Use -1 for any unknown labels
        
        # Convert label to a tensor
        label = torch.tensor(label, dtype=torch.long)
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

dataset = OcularDiseaseDataset(
    csv_path,
    train_path,
    data_transform
)
    
class RetinaDiseaseClassifier(nn.Module):
    def __init__(self, num_classes=8):
        super(RetinaDiseaseClassifier, self).__init__()
        # Where we define all the parts of the model
        self.base_model = timm.create_model('efficientnet_b0', pretrained=True)
        self.features = nn.Sequential(*list(self.base_model.children())[:-1])

        enet_out_size = 1280
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(enet_out_size, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        output = self.classifier(x)
        return output
    

# Calculate the lengths for the splits
total_size = len(dataset)
train_size = int(0.7 * total_size)
val_size = int(0.15 * total_size)
test_size = total_size - train_size - val_size  # Remaining data for testing

# Split the dataset into train, validation, and test sets
train_subset, val_subset, test_subset = random_split(dataset, [train_size, val_size, test_size])

# Create DataLoaders for each subset
train_loader = DataLoader(train_subset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_subset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_subset, batch_size=32, shuffle=False)

model = RetinaDiseaseClassifier(num_classes=8)

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