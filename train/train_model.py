import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.optim.lr_scheduler as lr_scheduler
import torchvision.transforms as transforms
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
import json
from datetime import datetime

from sys import path
path.append('../src')
from glaucoma_dataset import GlaucomaDataset
from glaucoma_model import GlaucomaDiagnoser

class ModelCheckpointer:
    def __init__(self, save_dir, model_name):
        self.save_dir = save_dir
        self.model_name = model_name
        self.best_val_loss = float('inf')
        self.best_epoch = -1
        os.makedirs(save_dir, exist_ok=True)
        
        # Create a log file
        self.log_file = os.path.join(save_dir, f"{model_name}_training_log.json")
        self.training_history = []

    def save_checkpoint(self, model, optimizer, scheduler, epoch, 
                       train_loss, val_loss, metrics=None, params=None):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'metrics': metrics,
        }

        # Save latest checkpoint
        latest_path = os.path.join(self.save_dir, f'{self.model_name}_latest.pth')
        torch.save(checkpoint, latest_path)

        # Save best model
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.best_epoch = epoch
            best_path = os.path.join(self.save_dir, f'{self.model_name}_best.pth')
            print(f'Saving best model val loss {self.best_val_loss}')
            torch.save(checkpoint, best_path)

        # Save periodic checkpoint
        if epoch % 10 == 0:
            periodic_path = os.path.join(self.save_dir, f'{self.model_name}_epoch_{epoch}.pth')
            torch.save(checkpoint, periodic_path)

        # Log training info
        log_entry = {
            'epoch': epoch,
            'train_loss': float(train_loss),
            'val_loss': float(val_loss),
            'learning_rate': optimizer.param_groups[0]['lr'],
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'metrics': metrics,
            'params': params
        }
        self.training_history.append(log_entry)
        
        with open(self.log_file, 'w') as f:
            json.dump(self.training_history, f, indent=4)

    def load_checkpoint(self, model, optimizer=None, scheduler=None, checkpoint_path=None):
        if checkpoint_path is None:
            checkpoint_path = os.path.join(self.save_dir, f'{self.model_name}_latest.pth')
        
        if not os.path.exists(checkpoint_path):
            print(f"No checkpoint found at {checkpoint_path}")
            return 0

        print('Loading checkpoint')
        checkpoint = torch.load(checkpoint_path, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        if optimizer and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if scheduler and 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
        return checkpoint['epoch']

class GlaucomaModelTrainer:
    def __init__(self, train_csvs, val_csvs, image_paths, checkpoint_path=None, 
                 finetune_params=None, **kwargs):
        self.train_csvs = train_csvs
        self.val_csvs = val_csvs
        self.image_paths = image_paths
        self.checkpoint_path = checkpoint_path
        
        # Default parameters
        self.default_params = {
            'batch_size': 32,
            'num_epochs': 100,
            'patience': 10,
            'base_model': 'efficientnet_b0',
            'lr': 0.0001,
            'wd': 1e-5,
            'dropout_rate': 0.2,
            'image_size': 256,
            'freeze_blocks': 0
        }
        
        # Update with provided parameters
        self.params = {**self.default_params, **kwargs}
        self.finetune_params = finetune_params or {}
        
        # Setup device and other attributes
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.train_loader = None
        self.val_loader = None
        self.model = None
        self.checkpointer = ModelCheckpointer(
            save_dir='model_checkpoints',
            model_name=f"glaucoma_{self.params['base_model']}",
        )

    def get_transforms(self, is_training=True):
        if is_training:
            return transforms.Compose([
                transforms.Resize((self.params['image_size'], self.params['image_size'])),
                transforms.RandomApply([
                    transforms.ColorJitter(brightness=0.3, contrast=0.3, 
                                        saturation=0.3, hue=0.1)
                ], p=0.7),
                transforms.RandomAffine(degrees=15, translate=(0.15, 0.15), 
                                      scale=(0.85, 1.15)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomApply([transforms.GaussianBlur(3)], p=0.4),
                transforms.ToTensor(),
                transforms.Normalize([0.21161936893269484, 0.0762942479204578, 0.02436706896535593], 
                                     [0.1694396363130937, 0.07732758785821338, 0.02954764478795169]),
                transforms.RandomErasing(p=0.3),
                transforms.RandomApply([
                    transforms.RandomAdjustSharpness(sharpness_factor=2)
                ], p=0.3),
            ])
        else:
            return transforms.Compose([
                transforms.Resize((self.params['image_size'], self.params['image_size'])),
                transforms.ToTensor(),
                transforms.Normalize([0.21161936893269484, 0.0762942479204578, 0.02436706896535593], 
                                     [0.1694396363130937, 0.07732758785821338, 0.02954764478795169])
            ])

    def setup_data(self):
        train_transform = self.get_transforms(is_training=True)
        val_transform = self.get_transforms(is_training=False)

        train_datasets, val_datasets = [], []

        for train_csv, image_path in zip(self.train_csvs, self.image_paths):
            train_datasets.append(GlaucomaDataset(train_csv, image_path, train_transform))

        for val_csv, image_path in zip(self.val_csvs, self.image_paths):
            val_datasets.append(GlaucomaDataset(val_csv, image_path, val_transform))

        train_dataset = torch.utils.data.ConcatDataset(train_datasets)
        val_dataset = torch.utils.data.ConcatDataset(val_datasets)

        # Setup weighted sampling for training
        train_labels = [train_dataset[i][1] for i in range(len(train_dataset))]
        class_counts = torch.bincount(torch.tensor(train_labels))
        weights = 1. / class_counts.float()
        sample_weights = weights[train_labels]
        sampler = torch.utils.data.WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True
        )

        self.train_loader = DataLoader(
            train_dataset, 
            batch_size=self.params['batch_size'],
            sampler=sampler,
            num_workers=4,
            pin_memory=True
        )
        
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=self.params['batch_size'],
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )

    def setup_model(self):
        self.model = GlaucomaDiagnoser(
            num_classes=2,
            base_model=self.params['base_model'],
            dropout_rate=self.params['dropout_rate'],
            freeze_blocks=self.params['freeze_blocks']
        ).to(self.device)

        if self.checkpoint_path:
            self.checkpointer.load_checkpoint(self.model, checkpoint_path=self.checkpoint_path)

    def train(self):
        print(f'Training {self.params["base_model"]} Starting')
        self.setup_data()
        self.setup_model()

        # Setup optimizer with differential learning rates
        base_model_params = list(self.model.base_model.parameters())
        classifier_params = list(self.model.classifier.parameters())
        
        optimizer = optim.AdamW([
            {'params': base_model_params, 'lr': self.params['lr'] * 0.1},
            {'params': classifier_params, 'lr': self.params['lr']}
        ], weight_decay=self.params['wd'])

        scheduler = lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=10, T_mult=2, eta_min=1e-6
        )

        criterion = nn.CrossEntropyLoss()
        scaler = torch.amp.GradScaler()
        
        best_val_loss = float('inf')
        patience_counter = 0

        for epoch in range(self.params['num_epochs']):
            # Training phase
            self.model.train()
            train_loss = 0
            
            for images, labels in tqdm(self.train_loader, desc=f'Epoch {epoch+1}'):
                images, labels = images.to(self.device), labels.to(self.device)
                
                optimizer.zero_grad()
                
                with torch.amp.autocast(device_type='cuda'):
                    outputs = self.model(images)
                    loss = criterion(outputs, labels)
                
                scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                
                train_loss += loss.item()

            train_loss = train_loss / len(self.train_loader)
            
            # Validation phase
            val_loss, metrics = self.validate()
            
            print(f'Epoch {epoch+1}:')
            print(f'Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
            print(f'Metrics: {metrics}')
            
            scheduler.step()
            
            # Save checkpoint
            self.checkpointer.save_checkpoint(
                model=self.model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                train_loss=train_loss,
                val_loss=val_loss,
                metrics=metrics,
                params=self.params
            )
            
            # Early stopping check
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.params['patience']:
                    print(f'Early stopping triggered at epoch {epoch+1} Best validation loss {best_val_loss}')
                    break

    def validate(self):
        self.model.eval()
        val_loss = 0
        correct = 0
        total = 0
        criterion = nn.CrossEntropyLoss()
        
        with torch.no_grad():
            for images, labels in self.val_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        accuracy = 100. * correct / total
        val_loss = val_loss / len(self.val_loader)
        
        metrics = {
            'accuracy': accuracy,
            'val_loss': val_loss
        }
        
        return val_loss, metrics

    def plot_training_history(self):
        with open(self.checkpointer.log_file, 'r') as f:
            history = json.load(f)
        
        epochs = [entry['epoch'] for entry in history]
        train_losses = [entry['train_loss'] for entry in history]
        val_losses = [entry['val_loss'] for entry in history]
        
        plt.figure(figsize=(10, 5))
        plt.plot(epochs, train_losses, label='Training Loss')
        plt.plot(epochs, val_losses, label='Validation Loss')
        plt.title('Training History')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.show()

def main():
    # Train model
    print('First training model')
    trainer = GlaucomaModelTrainer(
        train_csvs=['../data/dataset1/csvs/train.csv',
                    '../data/dataset2/csvs/train.csv'],
        val_csvs=['../data/dataset1/csvs/val.csv',
                  '../data/dataset2/csvs/val.csv'],
        image_paths=['../data/dataset1/processed',
                    '../data/dataset2/processed'],
        batch_size=32,
        num_epochs=100,
        patience=15,
        base_model='efficientnet_b0',
        lr=0.003,
        wd=0.05,
        image_size=256,
        freeze_blocks=7
    )
    trainer.train()
    trainer.plot_training_history()
    
    # Fine tune model
    print('Fine tuning model')
    finetuner = GlaucomaModelTrainer(
        train_csvs=['../data/dataset1/csvs/train.csv',
                    '../data/dataset2/csvs/train.csv'],
        val_csvs=['../data/dataset1/csvs/val.csv',
                  '../data/dataset2/csvs/val.csv'],
        image_paths=['../data/dataset1/processed',
                    '../data/dataset2/processed'],
        checkpoint_path='model_checkpoints/glaucoma_efficientnet_b0_best.pth', 
        batch_size=16,  
        num_epochs=50,  
        patience=10,
        base_model='efficientnet_b0', 
        lr=0.0001,  
        wd=0.01,
        image_size=256,
        freeze_blocks=6
    )
    finetuner.train()
    finetuner.plot_training_history()

if __name__ == '__main__':
    main()
