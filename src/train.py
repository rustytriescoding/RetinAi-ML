import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import matplotlib.pyplot as plt 
import numpy as np
import pandas as pd
import os
import cv2
from tqdm import tqdm

from GlaucomaDataset import GlaucomaDataset
from GlaucomaDiagnoser import GlaucomaDiagnoser

def calculate_mean_std(csv, image_path):
    mean = np.zeros(3)
    std = np.zeros(3)
    num_images = 0

    df = pd.read_csv(csv)

    for filename in tqdm(df['filename'], desc='Mean STD loop'):
        img_path = os.path.join(image_path, filename)
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        num_images += 1
        mean += img.mean(axis=(0, 1))
        std += img.std(axis=(0, 1))

    mean /= num_images
    std /= num_images

    return mean.tolist(), std.tolist()

def main():
    train_csv='../data/csvs/train.csv'
    val_csv='../data/csvs/val.csv'

    image_path='../data/ODIR-5K/Training Images'

    # Calculate Mean and Standard deviation of training images 
    # mean, std = calculate_mean_std(train_csv, image_path)
    # print(f'Mean: {mean}, Std: {std}')

    mean = [75.44004655992492/255, 47.15671788712379/255, 25.58542044342586/255]
    std = [69.76790132458073/255, 45.55243618334717/255, 26.263278919301026/255]

    IMAGE_SIZE = 224
    train_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)), 
        transforms.RandomApply([transforms.RandomHorizontalFlip()], p=0.2),
        transforms.RandomApply([transforms.RandomVerticalFlip()], p=0.2),
        transforms.RandomRotation(10), 
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    val_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)), 
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    train_dataset = GlaucomaDataset(
        train_csv,
        image_path,
        train_transform
    )
        
    val_dataset = GlaucomaDataset(
        val_csv,
        image_path,
        val_transform
    )

    # Batch size
    batch_size = 32
    # Create DataLoaders for each subset
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=False, num_workers=2) # See if pin memory makes it faster
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=False, num_workers=2)

    base_model='resnet50'

    model = GlaucomaDiagnoser(num_classes=2, base_model=base_model)

    # load model
    retinai_model_path=f'../models/{base_model}/retinai_{base_model}_0.0.1.pth'

    # model.load_state_dict(torch.load(retinai_model_path, weights_only=False))

    # Training loop
    num_epochs = 100
    train_losses, val_losses = [], []

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model.to(device)

    # Weights
    class_weights = torch.tensor([0.5492, 5.5841], dtype=torch.float32).to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-5)

    patience = 10
    best_val_loss = np.inf  
    patience_counter = 0 

    scaler = torch.amp.GradScaler()

    model_path = f'../models/{base_model}/'
    model_path = os.path.join(model_path, f'retinai_{base_model}_0.0.1.pth')

    for epoch in range(num_epochs):
        # Training
        model.train()
        running_loss = 0.0
        for images, labels in tqdm(train_loader, desc='Training loop'):
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            with torch.amp.autocast(device_type='cuda'):
                outputs = model(images)
                loss = criterion(outputs, labels)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item() * labels.size(0)
            
            # Free memory for each batch
            del images, labels, outputs, loss
            torch.cuda.empty_cache()

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

                # Free memory for each batch
                del images, labels, outputs, loss
                torch.cuda.empty_cache()

        val_loss = running_loss / len(val_loader.dataset)
        val_losses.append(val_loss)
        
        print(f"Epoch {epoch+1}/{num_epochs} - Train loss: {train_loss}, Validation loss: {val_loss}")

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss 
            patience_counter = 0
            if val_loss < 0.45597531341371084:
                torch.save(model.state_dict(), model_path)
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break


    
    # print('Saving model:', model_path)

    # Plot results
    plt.plot(train_losses, label='Training loss')
    plt.plot(val_losses, label='Validation loss')
    plt.legend()
    plt.title("Loss over epochs")
    plt.show()

if __name__ == '__main__':
    main()
