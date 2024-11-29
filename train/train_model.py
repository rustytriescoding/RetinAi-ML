import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.optim.lr_scheduler as lr_scheduler
import torchvision.transforms as transforms
import os
import matplotlib.pyplot as plt
import cv2
import numpy as np
from tqdm import tqdm
import json
from PIL import Image
from datetime import datetime

from sys import path
path.append('../src')
from glaucoma_dataset import GlaucomaDataset3
from glaucoma_model import GlaucomaDiagnoser
from checkpoint_model import ModelCheckpointer

def apply_bilateral_filter(img, d=9, sigma_color=75, sigma_space=75):
    img_np = np.array(img)
    
    filtered = cv2.bilateralFilter(img_np, d, sigma_color, sigma_space)
    
    return Image.fromarray(filtered)

def apply_clahe_to_green(img, clip_limit=2.0, tile_grid_size=(8, 8)):
    img_np = np.array(img)
    green = img_np[:, :, 1]
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    green_enhanced = clahe.apply(green)
    enhanced = np.stack([green_enhanced, green_enhanced, green_enhanced], axis=2)
    return Image.fromarray(enhanced.astype(np.uint8))

def apply_clahe(img, clip_limit=2.0, tile_grid_size=(8, 8)):
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    img_np = np.array(img)
    lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    l_clahe = clahe.apply(l)
    lab = cv2.merge((l_clahe, a, b))
    img_clahe = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    
    return Image.fromarray(img_clahe)

def extract_green_channel(x):
    # return x[1].unsqueeze(0).repeat(3, 1, 1)
    return x[1].unsqueeze(0)

class GlaucomaModelTrainer:
    def __init__(self, train_csvs, val_csvs, image_paths, checkpoint_path=None, 
                 finetune_params=None, **kwargs):
        self.train_csvs = train_csvs
        self.val_csvs = val_csvs
        self.image_paths = image_paths
        self.checkpoint_path = checkpoint_path
        self.class_weights = None
        
        # Default parameters
        self.default_params = {
            'batch_size': 32,
            'num_epochs': 100,
            'patience': 10,
            'base_model': 'resnet50',
            'lr': 0.0001, 
            'wd': 0.02,
            'dropout_rate': 0.6,
            'image_size': 224,
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
        base_transforms = [
            transforms.Resize((self.params['image_size'], self.params['image_size'])),
            # transforms.ToTensor(),
            # transforms.Lambda(lambda x: x[1].unsqueeze(0)),  # Extract green channel and keep single channel
            # transforms.Normalize(mean=[0.456], std=[0.224])  # Single channel normalization
        ]

        if is_training:
            base_transforms.extend([
                transforms.RandomHorizontalFlip(),
                # transforms.RandomVerticalFlip(),
                transforms.RandomRotation(
                    degrees=5,
                    fill=0
                ),
                transforms.RandomAffine(
                    degrees=0,
                    shear=(-20, 20),
                    scale=(0.8, 1.2),
                    fill=0
                ),
            ])

        base_transforms.extend([
            transforms.ToTensor(),
            transforms.Lambda(extract_green_channel), # Extract only green channel
            # transforms.Normalize(mean=[0.395, 0.395, 0.395], std=[0.181, 0.181, 0.181]) #green channel mean and std
            # transforms.Normalize(mean=[0.653, 0.395, 0.217], std=[0.231, 0.182, 0.147]) # Color all 3 channels mean and std
            # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) #image net mean and std
            transforms.Normalize(mean=[0.456], std=[0.224]) #image net mean and std
            # transforms.Normalize(mean=[0.456, 0.456, 0.456], std=[0.224, 0.224, 0.224]) #image net mean and std gren
        ])

        return transforms.Compose(base_transforms)

    def setup_data(self):
        train_transform = self.get_transforms(is_training=True)
        val_transform = self.get_transforms(is_training=False)

        train_datasets, val_datasets = [], []

        for train_csv, image_path in zip(self.train_csvs, self.image_paths):
            train_datasets.append(GlaucomaDataset3(
                train_csv, 
                image_path, 
                transform=train_transform,
            ))

        for val_csv, image_path in zip(self.val_csvs, self.image_paths):
            val_datasets.append(GlaucomaDataset3(
                val_csv, 
                image_path, 
                transform=val_transform,
            ))

        train_dataset = torch.utils.data.ConcatDataset(train_datasets)
        val_dataset = torch.utils.data.ConcatDataset(val_datasets)

        print('Train Dataset Length', len(train_dataset))
        print('Validation Dataset Length', len(val_dataset))

        # train_labels = [int(train_dataset[i][1].item()) for i in range(len(train_dataset))]
        # class_counts = torch.bincount(torch.tensor(train_labels))
        # print('Weight:', class_counts[0] / class_counts[1])
        # self.class_weights = torch.tensor([class_counts[0] / class_counts[1]]).to(self.device)
        self.class_weights = torch.tensor([1.6209]).to(self.device) # Hardcode weights to reduce compute time [7.26]

        # print(f"Class counts - Normal: {class_counts[0]}, Glaucoma: {class_counts[1]}")
        print(f"Using positive class weight: {self.class_weights.item():.2f}") 
        

        self.train_loader = DataLoader(
            train_dataset, 
            batch_size=self.params['batch_size'],
            shuffle=True,
            num_workers=2,
            pin_memory=True
        )
        
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=self.params['batch_size'],
            shuffle=False,
            num_workers=2,
            pin_memory=True
        )

    def setup_model(self):
        self.model = GlaucomaDiagnoser(
            base_model=self.params['base_model'],
            dropout_rate=self.params['dropout_rate'],
            freeze_blocks=self.params['freeze_blocks'],
        ).to(self.device)

        if self.checkpoint_path:
            self.checkpointer.load_checkpoint(self.model, checkpoint_path=self.checkpoint_path)

    def train(self):
        print(f'Training {self.params["base_model"]} Starting')
        self.setup_data()
        self.setup_model()

        criterion = nn.BCEWithLogitsLoss(pos_weight=self.class_weights)
        # criterion = nn.BCEWithLogitsLoss()
        
        params = [
            {'params': self.model.model.parameters(), 'lr': self.params['lr']},
            {'params': self.model.classifier.parameters(), 'lr': self.params['lr']},
        ]
        
        optimizer = optim.AdamW(params, weight_decay=self.params['wd'])
       
        
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.9, # 0.2
            patience=2,   # 3
            min_lr=1e-6
        )

        scaler = torch.amp.GradScaler()
        best_val_loss = float('inf')
        best_val_f1 = 0
        patience_counter = 0
        
        for epoch in range(self.params['num_epochs']):
            # Training phase
            self.model.train()
            train_loss = 0
            train_predictions = []
            train_labels_list = []
            
            for full_images, labels in tqdm(self.train_loader, desc='Training Loop'):
                full_images = full_images.to(self.device)
                labels = labels.to(self.device).float()
                
                optimizer.zero_grad()
                
                with torch.amp.autocast(device_type='cuda'):
                    outputs = self.model(full_images)
                    outputs = outputs.squeeze(1)
                    loss = criterion(outputs, labels)

                scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                scaler.step(optimizer)
                scaler.update()
                
                train_loss += loss.item()
                
                # Calculate predictions
                predicted = (torch.sigmoid(outputs) > 0.5).float()
                train_predictions.extend(predicted.cpu().numpy())
                train_labels_list.extend(labels.cpu().numpy())
            
            train_loss = train_loss / len(self.train_loader)
            
            train_f1 = self.calculate_f1_score(train_predictions, train_labels_list)
            
            # Validation phase
            val_loss, metrics = self.validate()
            val_f1 = metrics['f1']
            scheduler.step(val_loss)
            
            print(f'Epoch {epoch+1}:')
            print(f'Train Loss: {train_loss:.4f}, Train F1: {train_f1:.4f}')
            print(f'Val Loss: {val_loss:.4f}, Val F1: {metrics["f1"]:.4f}')
            
            # Save checkpoint
            self.checkpointer.save_checkpoint(
                model=self.model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                train_loss=train_loss,
                val_loss=val_loss,
                val_f1=val_f1,
                metrics=metrics,
                params=self.params
            )
            
            # Early stopping check
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            # Early stopping check using F1 score
            # if val_f1 > best_val_f1:
            #     best_val_f1 = val_f1
            #     patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.params['patience']:
                    print(f'Early stopping triggered at epoch {epoch+1} Best F1 {best_val_f1} Best Val Loss {best_val_loss}')
                    break
    

    def validate(self):
        self.model.eval()
        val_loss = 0
        val_predictions = []
        val_labels_list = []
        criterion = nn.BCEWithLogitsLoss(pos_weight=self.class_weights)
        # criterion = nn.BCEWithLogitsLoss()
        
        with torch.no_grad():
            for full_images, labels in tqdm(self.val_loader, desc='Validation Loop'):
                full_images = full_images.to(self.device)
                labels = labels.to(self.device).float()
                
                outputs = self.model(full_images)
                outputs = outputs.squeeze(1)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
                predicted = (torch.sigmoid(outputs) > 0.5).float()
                val_predictions.extend(predicted.cpu().numpy())
                val_labels_list.extend(labels.cpu().numpy())

        val_loss = val_loss / len(self.val_loader)
        val_f1 = self.calculate_f1_score(val_predictions, val_labels_list)
        
        metrics = {
            'f1': val_f1,
            'val_loss': val_loss
        }
        
        return val_loss, metrics
    
    def calculate_f1_score(self, predictions, labels):
        predictions = np.array(predictions)
        labels = np.array(labels)
        
        true_positives = np.sum((predictions == 1) & (labels == 1))
        predicted_positives = np.sum(predictions == 1)
        actual_positives = np.sum(labels == 1)
        
        precision = true_positives / predicted_positives if predicted_positives > 0 else 0
        recall = true_positives / actual_positives if actual_positives > 0 else 0
        
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        return f1

    def plot_training_history(self, log_file=None):
        results_dir = 'train_results'
        os.makedirs(results_dir, exist_ok=True)
        
        # Load history
        if log_file:
            with open(log_file, 'r') as f:
                history = json.load(f)
        else:
            with open(self.checkpointer.log_file, 'r') as f:
                history = json.load(f)
        
        fig = plt.figure(figsize=(20, 10)) 
        
        gs = plt.GridSpec(2, 2, width_ratios=[2, 1], height_ratios=[1, 1], hspace=0.3)
        
        # Loss plot (top left)
        ax1 = fig.add_subplot(gs[0, 0])
        epochs = [entry['epoch'] for entry in history]
        train_losses = [entry['train_loss'] for entry in history]
        val_losses = [entry['val_loss'] for entry in history]
        
        ax1.plot(epochs, train_losses, label='Training Loss')
        ax1.plot(epochs, val_losses, label='Validation Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # F1 score plot (bottom left)
        ax2 = fig.add_subplot(gs[1, 0])
        f1_scores = [entry['metrics']['f1'] for entry in history]
        
        ax2.plot(epochs, f1_scores, label='Validation F1 Score', color='green')
        ax2.set_title('Validation F1 Score')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('F1 Score')
        ax2.legend()
        ax2.grid(True)
        
        # Parameters text (right side, spans both rows)
        ax3 = fig.add_subplot(gs[:, 1])
        ax3.axis('off')
        
        # Get final metrics and parameters from history
        final_metrics = history[-1].get('metrics', {})
        params = history[-1].get('params', {})
        
        # Create parameter text
        param_text = "Training Parameters:\n"
        param_text += f"Model: {params.get('base_model', 'N/A')}\n"
        param_text += f"Learning Rate: {params.get('lr', 'N/A')}\n"
        param_text += f"Weight Decay: {params.get('wd', 'N/A')}\n"
        param_text += f"Batch Size: {params.get('batch_size', 'N/A')}\n"
        param_text += f"Dropout Rate: {params.get('dropout_rate', 'N/A')}\n"
        param_text += f"Image Size: {params.get('image_size', 'N/A')}\n"
        param_text += f"Frozen Blocks: {params.get('freeze_blocks', 'N/A')}\n"
        
        param_text += "Final Results:\n"
        param_text += f"Best Validation Loss: {min([entry['val_loss'] for entry in history]):.4f}\n"
        param_text += f"Final Validation Loss: {history[-1]['val_loss']:.4f}\n"
        param_text += f"Best F1 Score: {max([entry['metrics']['f1'] for entry in history]):.4f}\n"
        param_text += f"Final F1 Score: {final_metrics.get('f1', 'N/A'):.4f}\n"
        param_text += f"Total Epochs: {len(epochs)}\n"
        
        ax3.text(0.1, 0.9, param_text, fontsize=10, verticalalignment='top', 
                family='monospace')
        
        plt.tight_layout()
        
        # Create unique filename with timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        model_name = params.get('base_model', 'unknown_model')
        filename = f"{results_dir}/{model_name}_{timestamp}.png"
        
        # Save figure
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Training results saved to {filename}")
        
        # Save parameters and metrics to JSON 
        results_dict = {
            'parameters': params,
            'training_history': history,
            'final_metrics': final_metrics,
        }
        
        json_filename = f"{results_dir}/{model_name}_{timestamp}.json"
        with open(json_filename, 'w') as f:
            json.dump(results_dict, f, indent=4)
        
        plt.show()

def main():
   
    # Best Val F1 0.8411 Val Loss 0.3699 Using resnet18, training at 0.0001 to failure, then trying again with 0.001
    # print('--- Phase 1 ---')
    # trainer = GlaucomaModelTrainer(
    #     train_csvs=['../data/dataset3/csvs/train.csv'],
    #     val_csvs=['../data/dataset3/csvs/val.csv'],
    #     image_paths=['../data/dataset3/disc-crop'], 
    #     num_epochs=10,
    #     patience=2,
    #     base_model='efficientnet_b0',
    #     wd=0.03, 
    #     image_size=224,
    #     freeze_blocks=6,
    #     dropout_rate=0.2,
    #     lr=0.0001, 
    # )
    # trainer.train()

    # print('--- Phase 2 ---')
    # trainer = GlaucomaModelTrainer(
    #     train_csvs=['../data/dataset3/csvs/train.csv'],
    #     val_csvs=['../data/dataset3/csvs/val.csv'],
    #     image_paths=['../data/dataset3/disc-crop'], 
    #     num_epochs=100,
    #     patience=2,
    #     base_model='efficientnet_b0',
    #     wd=0.03, 
    #     image_size=224,
    #     freeze_blocks=3,
    #     dropout_rate=0.5,
    #     lr=0.0001,
    #     checkpoint_path='model_checkpoints/glaucoma_efficientnet_b0_latest.pth' 
    # )
    # trainer.train()

    # print('--- Phase 3 ---')
    # trainer = GlaucomaModelTrainer(
    #     train_csvs=['../data/dataset3/csvs/train.csv'],
    #     val_csvs=['../data/dataset3/csvs/val.csv'],
    #     image_paths=['../data/dataset3/disc-crop'], 
    #     num_epochs=100,
    #     patience=5,
    #     base_model='efficientnet_b0',
    #     wd=0.05, 
    #     image_size=224,
    #     freeze_blocks=0,
    #     dropout_rate=0.6,
    #     lr=0.0002, 
    #     checkpoint_path='model_checkpoints/glaucoma_efficientnet_b0_latest.pth' 
    # )
    # trainer.train()
    # trainer.plot_training_history()

    trainer = GlaucomaModelTrainer(
        train_csvs=['../data/dataset3/csvs/train.csv', '../data/dataset4/csvs/train.csv'],
        val_csvs=['../data/dataset3/csvs/val.csv', '../data/dataset4/csvs/val.csv'],
        image_paths=['../data/dataset3/disc-crop', '../data/dataset4/disc-crop'], 
        num_epochs=100,
        patience=5,
        base_model='efficientnet_b0',
        wd=0.2, 
        image_size=224,
        freeze_blocks=1,
        dropout_rate=0.6,
        lr=0.0001, 
    )
    trainer.train()
    trainer.plot_training_history()

if __name__ == '__main__':
    main()
    