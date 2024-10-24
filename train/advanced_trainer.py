import torch
from torch.utils.data import ConcatDataset, DataLoader
import numpy as np
from sklearn.model_selection import KFold
from copy import deepcopy

from sys import path
path.append('../src')
from glaucoma_dataset import GlaucomaDataset
from train.train_model import GlaucomaModelTrainer

class AdvancedGlaucomaTrainer:
    def __init__(self, base_trainer):
        self.base_trainer = base_trainer
        self.models = []
        self.best_val_acc = 0
        
    def train_with_cross_validation(self, k_folds=5):
        train_dataset = GlaucomaDataset(
            self.base_trainer.train_csv, 
            self.base_trainer.image_path,
            self.base_trainer.get_transforms(is_training=True)
        )
        val_dataset = GlaucomaDataset(
            self.base_trainer.val_csv,
            self.base_trainer.image_path,
            self.base_trainer.get_transforms(is_training=True)
        )
        
        full_dataset = ConcatDataset([train_dataset, val_dataset])
        kfold = KFold(n_splits=k_folds, shuffle=True, random_state=42)
        
        for fold, (train_ids, val_ids) in enumerate(kfold.split(full_dataset)):
            print(f'Training Fold {fold + 1}/{k_folds}')
            
            fold_trainer = deepcopy(self.base_trainer)
            
            train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
            val_subsampler = torch.utils.data.SubsetRandomSampler(val_ids)
            
            fold_trainer.train_loader = DataLoader(
                full_dataset,
                batch_size=self.base_trainer.params['batch_size'],
                sampler=train_subsampler,
                num_workers=4,
                pin_memory=True
            )
            
            fold_trainer.val_loader = DataLoader(
                full_dataset,
                batch_size=self.base_trainer.params['batch_size'],
                sampler=val_subsampler,
                num_workers=4,
                pin_memory=True
            )
            
            fold_trainer.train()
            
            _, metrics = fold_trainer.validate()
            if metrics['accuracy'] > self.best_val_acc:
                self.best_val_acc = metrics['accuracy']
                torch.save(fold_trainer.model.state_dict(), 
                         f'model_checkpoints/best_fold_{fold}.pth')
            
            self.models.append(fold_trainer.model)
    
    def ensemble_predict(self, test_loader):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(device)
                ensemble_preds = []
                
                # Get predictions from each model
                for model in self.models:
                    model.eval()
                    outputs = model(images)
                    probs = torch.softmax(outputs, dim=1)
                    ensemble_preds.append(probs.cpu().numpy())
                
                # Average predictions
                avg_preds = np.mean(ensemble_preds, axis=0)
                pred_labels = np.argmax(avg_preds, axis=1)
                
                all_predictions.extend(pred_labels)
                all_labels.extend(labels.numpy())
        
        return np.array(all_predictions), np.array(all_labels)

def train_advanced_model():
    base_trainer = GlaucomaModelTrainer(
        train_csv='../data/csvs/train.csv',
        val_csv='../data/csvs/val.csv',
        image_path='../data/Cropped Images',
        batch_size=16,
        num_epochs=50,
        patience=10,
        base_model='efficientnet_b2',  
        lr=0.0005,
        wd=0.01,
        image_size=380,
        dropout_rate=0.3
    )
    
    advanced_trainer = AdvancedGlaucomaTrainer(base_trainer)
    
    advanced_trainer.train_with_cross_validation(k_folds=5)
    
    test_transform = base_trainer.get_transforms(is_training=False)
    test_dataset = GlaucomaDataset(
        '../data/csvs/test.csv',
        '../data/Cropped Images',
        test_transform
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=4
    )
    
    predictions, labels = advanced_trainer.ensemble_predict(test_loader)
    
    accuracy = np.mean(predictions == labels) * 100
    print(f'Ensemble Model Test Accuracy: {accuracy:.2f}%')

if __name__ == '__main__':
    train_advanced_model()
