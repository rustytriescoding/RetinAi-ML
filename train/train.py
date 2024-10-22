import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import matplotlib.pyplot as plt 
import numpy as np
import os
from tqdm import tqdm

from sys import path
path.append('../src')
from GlaucomaDataset import GlaucomaDataset, GlaucomaDataset2
from GlaucomaDiagnoser import GlaucomaDiagnoser

class GlaucomaModelTrainer:
  def __init__(self, train_csv, val_csv, image_path, batch_size=32, num_epochs=100, patience=10, base_model='efficientnet_b0', model_path=None, lr=0.0001, wd=1e-5):
        self.train_csv = train_csv
        self.val_csv = val_csv
        self.image_path = image_path
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.patience = patience
        self.base_model = base_model
        self.model_path = model_path
        self.lr = lr
        self.wd = wd
        
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.mean = [75.44004655992492/255, 47.15671788712379/255, 25.58542044342586/255]
        self.std = [69.76790132458073/255, 45.55243618334717/255, 26.263278919301026/255]
        self.weight = [[0.5492, 5.5841], [0.6736, 1.9403]]
        self.train_loader = None
        self.val_loader = None
        self.train_losses = []
        self.val_losses = []
        self.model = None

  def loadData(self):
    # Define image size
    IMAGE_SIZE = 224

    # Define transforms
    train_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)), 
        transforms.RandomApply([transforms.RandomHorizontalFlip()], p=0.2),
        transforms.RandomApply([transforms.RandomVerticalFlip()], p=0.2),
        transforms.RandomRotation(10), 
        transforms.ToTensor(),
        # GammaAdjust(gamma=0.4),
        transforms.Normalize(self.mean, self.std)
    ])
    val_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)), 
        transforms.ToTensor(),
        # GammaAdjust(gamma=0.4),
        transforms.Normalize(self.mean, self.std),
    ])

    # Define datasets
    train_dataset = GlaucomaDataset(
        self.train_csv,
        self.image_path,
        train_transform
    )
        
    val_dataset = GlaucomaDataset(
        self.val_csv,
        self.image_path,
        val_transform
    )

    # Load data
    self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, pin_memory=False, num_workers=2)
    self.val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False, pin_memory=False, num_workers=2)

  def loadModel(self):
    self.model = GlaucomaDiagnoser(num_classes=2, base_model=self.base_model)
    if self.model_path: self.model.load_state_dict(torch.load(self.model_path, weights_only=False))
    self.model.to(self.device)

  def train(self):
    self.loadData()
    self.loadModel()

    class_weights = torch.tensor(self.weight[0], dtype=torch.float32).to(self.device)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.wd)

    best_val_loss = np.inf 
    patience_counter = 0 

    scaler = torch.amp.GradScaler()

    model_path = f'../models/{self.base_model}/'
    model_save_path = os.path.join(model_path, f'retinai_{self.base_model}_0.0.1.pth')

    for epoch in range(self.num_epochs):
        # Training
        self.model.train()
        running_loss = 0.0
        for images, labels in tqdm(self.train_loader, desc='Training loop'):
            images, labels = images.to(self.device), labels.to(self.device)
            
            optimizer.zero_grad()
            
            with torch.amp.autocast(device_type='cuda'):
                outputs = self.model(images)
                loss = criterion(outputs, labels)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item() * labels.size(0)
            
            # Free memory for each batch
            del images, labels, outputs, loss
            torch.cuda.empty_cache()

        train_loss = running_loss / len(self.train_loader.dataset)
        self.train_losses.append(train_loss)

        # Validation
        self.model.eval()
        running_loss = 0.0
        with torch.no_grad():
            for images, labels in tqdm(self.val_loader, desc='Validation loop'):
                images, labels = images.to(self.device), labels.to(self.device)
                
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                running_loss += loss.item() * labels.size(0)

                # Free memory for each batch
                del images, labels, outputs, loss
                torch.cuda.empty_cache()

        val_loss = running_loss / len(self.val_loader.dataset)
        self.val_losses.append(val_loss)
        
        print(f"Epoch {epoch+1}/{self.num_epochs} - Train loss: {train_loss}, Validation loss: {val_loss}")

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss 
            patience_counter = 0
            if val_loss < 0.5:
                print('Saving model:', model_save_path)
                if not os.path.exists(model_path): os.makedirs(model_path)
                torch.save(self.model.state_dict(), model_save_path)
        else:
            patience_counter += 1

        if patience_counter >= self.patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

  def plotResults(self):
    # Plot training results
    plt.plot(self.train_losses, label='Training loss')
    plt.plot(self.val_losses, label='Validation loss')
    plt.legend()
    plt.title("Loss over epochs")
    plt.show()

def main():
  trainer = GlaucomaModelTrainer(
    train_csv='../data/csvs/train.csv',
    val_csv='../data/csvs/val.csv',
    image_path='../data/ODIR-5K/Training Images',
    batch_size=32,
    num_epochs=100,
    patience=10,
  )
  trainer.train()
  trainer.plotResults()

if __name__ == '__main__':
  main()
