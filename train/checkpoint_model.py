import torch
import os
import json
from datetime import datetime

from sys import path
path.append('../src')

class ModelCheckpointer:
    def __init__(self, save_dir, model_name):
        self.save_dir = save_dir
        self.model_name = model_name
        self.best_val_loss = float('inf')
        self.best_val_f1 = 0
        self.best_epoch = -1
        os.makedirs(save_dir, exist_ok=True)
        
        # Create a log file
        self.log_file = os.path.join(save_dir, f"{model_name}_training_log.json")
        self.training_history = []

    def save_checkpoint(self, model, optimizer, epoch, 
                       train_loss, val_loss, val_f1, metrics=None, scheduler=None, params=None):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'metrics': metrics,
            'f1': val_f1
        }

        # Save latest checkpoint
        latest_path = os.path.join(self.save_dir, f'{self.model_name}_latest.pth')
        torch.save(checkpoint, latest_path)

        # Save best model
        # if val_f1 > self.best_val_f1:
        if val_loss < self.best_val_loss:
            # self.best_val_f1 = val_f1
            self.best_val_loss = val_loss
            self.best_epoch = epoch
            best_path = os.path.join(self.save_dir, f'{self.model_name}_best.pth')
            print(f'Saving best model Val F1 {val_f1:.4f} Val Loss {self.best_val_loss:.4f}')
            torch.save(checkpoint, best_path)

        # Log training info
        log_entry = {
            'epoch': epoch,
            'train_loss': float(train_loss),
            'val_loss': float(val_loss),
            'learning_rate': optimizer.param_groups[0]['lr'],
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'metrics': metrics,
            'params': params,
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
