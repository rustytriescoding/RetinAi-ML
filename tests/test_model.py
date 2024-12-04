import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import json
from datetime import datetime
import os

from sys import path
path.append('../src')
from glaucoma_dataset import GlaucomaDataset, GlaucomaDataset3
from glaucoma_model import GlaucomaDiagnoser

def extract_green_channel(x):
    return x[1:2, :, :] 

class GlaucomaModelTester:
    def __init__(self, model_path, image_path, test_csvs, base_model='resnet50', 
                 image_size=224, batch_size=32):
        self.model_path = model_path
        self.image_path = image_path
        self.test_csvs = test_csvs
        self.base_model = base_model
        self.image_size = image_size
        self.batch_size = batch_size
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        # Create results directory
        self.results_dir = 'test_results'
        os.makedirs(self.results_dir, exist_ok=True)
        
        self.setup_model()
        self.setup_transforms()

    def setup_model(self):
        self.model = GlaucomaDiagnoser(base_model=self.base_model)
        checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()

    def setup_transforms(self):
        self.transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            transforms.Lambda(extract_green_channel), 
            # transforms.Normalize(mean=[0.395, 0.395, 0.395], std=[0.182, 0.182, 0.182]) 
            transforms.Normalize(mean=[0.456], std=[0.224])
        ])

    def test_model(self, csv_path):
        dataset = GlaucomaDataset3(
        # dataset = GlaucomaDataset(
            csv_path, 
            self.image_path, 
            self.transform,
        )
        
        loader = DataLoader(
            dataset, 
            batch_size=self.batch_size, 
            shuffle=False, 
            num_workers=4,
            pin_memory=True
        )

        all_predictions = []
        all_labels = []
        all_probabilities = []

        with torch.no_grad():
            for full_images, labels in tqdm(loader, desc=f"Testing {os.path.basename(csv_path)}"):
                full_images = full_images.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(full_images)
                outputs = outputs.squeeze(1)  # Remove the extra dimension
                probabilities = torch.sigmoid(outputs)  # Use sigmoid for binary classification
                predicted = (outputs > 0).float()  # Convert logits to binary predictions

                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())

        return np.array(all_predictions), np.array(all_labels), np.array(all_probabilities)

    def plot_confusion_matrix(self, y_true, y_pred, title, save_path):
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['Normal', 'Glaucoma'],
                   yticklabels=['Normal', 'Glaucoma'])
        plt.title(f'Confusion Matrix - {title}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig(save_path)
        plt.close()

    def plot_roc_curve(self, y_true, y_prob, title, save_path):
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_auc = auc(fpr, tpr)

        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2,
                label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - {title}')
        plt.legend(loc="lower right")
        plt.savefig(save_path)
        plt.close()

    def save_metrics(self, metrics, filename):
        with open(os.path.join(self.results_dir, filename), 'w') as f:
            json.dump(metrics, f, indent=4)

    def run_tests(self):
        test_results = {}
        
        for csv_path in self.test_csvs:
            test_name = os.path.basename(csv_path).split('.')[0]
            print(f'Starting testing {self.model_path}')
            
            # Run predictions
            predictions, labels, probabilities = self.test_model(csv_path)
            
            # Calculate metrics
            report = classification_report(labels, predictions, target_names=['Normal', 'Glaucoma'], 
                                        output_dict=True)
            
            # Save confusion matrix
            self.plot_confusion_matrix(
                labels, predictions,
                test_name,
                os.path.join(self.results_dir, f'confusion_matrix_{test_name}.png')
            )
            
            # Save ROC curve
            self.plot_roc_curve(
                labels, probabilities,
                test_name,
                os.path.join(self.results_dir, f'roc_curve_{test_name}.png')
            )
            
            # Calculate additional metrics
            accuracy = (predictions == labels).mean() * 100
            specificity = report['Normal']['recall']
            sensitivity = report['Glaucoma']['recall']
            
            # Store results
            test_results[test_name] = {
                'accuracy': accuracy,
                'specificity': specificity,
                'sensitivity': sensitivity,
                'classification_report': report,
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            
            # Print results
            print(f"Accuracy: {accuracy:.2f}%")
            print(f"Specificity: {specificity:.2f}")
            print(f"Sensitivity: {sensitivity:.2f}")
            print("\nClassification Report:")
            print(classification_report(labels, predictions, target_names=['Normal', 'Glaucoma']))
        
        # Save all results
        self.save_metrics(test_results, 'test_results.json')
        return test_results

def main():
    tester = GlaucomaModelTester(
        # model_path='../models/mobilenetv3/mobilenetv3_94.pth',
        model_path='../train/model_checkpoints/glaucoma_tf_mobilenetv3_small_100_best.pth',
        # model_path='../train/model_checkpoints/glaucoma_resnet50_best.pth',
        # model_path='../models/resnet50/resnet50_89.pth',
        image_path='../data/dataset4/disc-crop', 
        test_csvs=['../data/dataset4/csvs/test.csv',],
        # base_model='resnet50',
        base_model='tf_mobilenetv3_small_100',
        # image_size=256,
        image_size=224,
        batch_size=32
    )
    
    tester.run_tests()

if __name__ == '__main__':
    main()
