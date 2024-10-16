import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import timm

import matplotlib.pyplot as plt # For data viz
import pandas as pd
import numpy as np
import sys
import os
from tqdm.notebook import tqdm

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

train_dataset = OcularDiseaseDataset(
    csv_path,
    train_path,
    data_transform
)

image, label = train_dataset[16]

train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)

for images, labels in train_dataloader:
    break
    
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
    
model = RetinaDiseaseClassifier(num_classes=8)
print(str(model)[:500])