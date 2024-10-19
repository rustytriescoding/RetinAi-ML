import torch
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm
import torchvision.transforms as transforms
from GlaucomaDataset import GlaucomaDataset
from GlaucomaDiagnoser import GlaucomaDiagnoser

test_csv = '../data/csvs/test.csv'
image_path = '../data/ODIR-5K/Training Images'

IMAGE_SIZE = 224
data_transform = transforms.Compose([transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)), transforms.ToTensor()])

test_dataset = GlaucomaDataset(
    test_csv,
    image_path,
    data_transform
)

test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

model = GlaucomaDiagnoser(num_classes=3, base_model='resnet18')

retinai_resnet50_path='../models/resnet50/retinai_resnet50_0.0.1.pth'
retinai_resnet18_path='../models/resnet18/retinai_resnet18_0.0.1.pth'
model.load_state_dict(torch.load(retinai_resnet18_path, weights_only=False))
model.eval()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

correct = 0
total = 0
predictions = []
actuals = []

with torch.no_grad():
    for images, labels in tqdm(test_loader, desc="Testing"):
        images, labels = images.to(device), labels.to(device)
        
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)

        predictions.extend(predicted.cpu().numpy())
        actuals.extend(labels.cpu().numpy())

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f'Accuracy of the model on the test dataset: {accuracy:.2f}%')
