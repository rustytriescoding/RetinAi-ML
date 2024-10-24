import torch
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm
import torchvision.transforms as transforms

from sys import path
path.append('../src')
from glaucoma_dataset import GlaucomaDataset
from glaucoma_model import GlaucomaDiagnoser

# mean = [0.485, 0.456, 0.406]
# std = [0.229, 0.224, 0.225]
mean = [0.21161936893269484, 0.0762942479204578, 0.02436706896535593]
std = [0.1694396363130937, 0.07732758785821338, 0.02954764478795169]

base_model = 'efficientnet_b0'
model = GlaucomaDiagnoser(num_classes=2, base_model=base_model)

checkpoint = torch.load('../train/model_checkpoints/glaucoma_efficientnet_b0_best.pth', weights_only=False)
model.load_state_dict(checkpoint['model_state_dict'])
    
model.eval()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

def main():
    csvs = [
        '../data/csvs/test5050.csv',
        '../data/csvs/testonly-N.csv',
        '../data/csvs/testonly-G.csv'
    ]
    
    image_path = '../data/Cropped Images'

    IMAGE_SIZE = 256
    data_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)), 
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    for csv in csvs:
        test_dataset = GlaucomaDataset(
            csv,
            image_path,
            data_transform
        )

        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, pin_memory=False, num_workers=2)

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
        print(f'Accuracy of the model on the {csv} dataset: {accuracy:.2f}% ')

if __name__ == '__main__':
    main()
