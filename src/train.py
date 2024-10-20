import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import matplotlib.pyplot as plt 
import numpy as np
import os
from tqdm import tqdm

from GlaucomaDataset import GlaucomaDataset
from GlaucomaDiagnoser import GlaucomaDiagnoser

def main():
    train_csv='../data/csvs/train.csv'
    val_csv='../data/csvs/val.csv'

    image_path='../data/ODIR-5K/Training Images'

    IMAGE_SIZE = 224
    train_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)), 
        transforms.RandomApply([transforms.RandomHorizontalFlip()], p=0.2),
        transforms.RandomApply([transforms.RandomVerticalFlip()], p=0.2),
        transforms.RandomRotation(10), 
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229]),
    ])

    val_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)), 
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229]),
    ])

    diagnosis_list = ['Normal', 'Diabetes', 'Glaucoma', 'Cataract', 'Age', 'Hypertension', 'Pathological Myopia', 'Other']

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

    # Create DataLoaders for each subset
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, pin_memory=False, num_workers=6)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, pin_memory=False, num_workers=6)

    base_model='efficientnet_b0'

    model = GlaucomaDiagnoser(num_classes=2, base_model=base_model)

    # load model
    retinai_model_path=f'../models/{base_model}/retinai_{base_model}_0.0.1.pth'

    # model.load_state_dict(torch.load(retinai_model_path, weights_only=False))

    # Simple training loop
    num_epochs = 50
    train_losses, val_losses = [], []

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-5)

    patience = 3
    best_val_loss = np.inf  
    patience_counter = 0 

    scaler = torch.amp.GradScaler()

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
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break


    model_path = f'../models/{base_model}/'
    model_path = os.path.join(model_path, f'retinai_{base_model}_0.0.1.pth')
    torch.save(model.state_dict(), model_path)
    print('Saving model:', model_path)

    plt.plot(train_losses, label='Training loss')
    plt.plot(val_losses, label='Validation loss')
    plt.legend()
    plt.title("Loss over epochs")
    plt.show()

if __name__ == '__main__':
    main()
