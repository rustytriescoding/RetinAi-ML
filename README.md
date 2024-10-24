# Glaucoma Detection Project

This repository contains a deep learning-based solution for automated glaucoma detection from retinal images. The project utilizes transfer learning with EfficientNet architectures to classify retinal images as either normal or showing signs of glaucoma.

## 🎯 Features

- Transfer learning using EfficientNet models
- Support for multiple datasets
- Weighted sampling for handling class imbalance
- Mixed precision training
- Checkpoint management and model versioning
- Comprehensive testing and evaluation metrics
- Multi-dataset training support

## 🔧 Project Structure

```
.
├── .vscode/                  # VSCode configuration
├── data/
│   ├── dataset1/
│   │   ├── csvs/            # Data annotations
│   │   ├── processed/       # Preprocessed images
│   │   └── raw/            # Original raw images
│   ├── dataset2/
│   │   ├── csvs/
│   │   ├── processed/
│   │   └── raw/
│   └── scripts/            # Data preprocessing and formatting scripts
├── models/                 # Saved model weights and configurations
├── src/
│   ├── pycache/
│   ├── glaucoma_dataset.py # Dataset loader and preprocessing
│   └── glaucoma_model.py   # Model architecture definition
├── tests/
│   ├── test_results/      # Test output and metrics
│   ├── diagnosis.py
│   ├── test_model.py
│   └── test.py
├── train/
│   ├── model_checkpoints/ # Training checkpoints
│   ├── advanced_trainer.py
│   └── train_model.py     # Training script
├── venv/                  # Python virtual environment
├── .gitignore
├── README.md
└── requirements.txt      # Project dependencies
```

## 📋 Requirements

- Python 3.7+
- PyTorch 1.7+
- torchvision
- timm
- pandas
- numpy
- Pillow
- scikit-learn
- matplotlib
- seaborn
- tqdm

## 🚀 Getting Started

1. **Setup Environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use: venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. **Prepare Dataset**
   - Place your raw retinal images in the `data/dataset*/raw/` directories
   - Run preprocessing scripts from `data/scripts/` to process images
   - Processed images will be saved to `data/dataset*/processed/`
   - Create CSV files with image annotations in `data/dataset*/csvs/`
   - CSV format should include either:
     - 'filename' and 'labels' columns (with 'N'/'G' labels)
     - 'Filename' and 'Glaucoma' columns (with 0/1 labels)

3. **Training**
   ```bash
   cd train
   py train_model.py
   ```

4. **Testing**
   ```bash
   cd tests
   python test_model.py
   ```

## 💡 Model Architecture

The model uses a pretrained EfficientNet B0 backbone with a custom classifier head:
- Backbone: EfficientNet (B0-B7 supported)
- Custom classifier with:
  - Dropout layers for regularization
  - Batch normalization
  - ReLU activation
  - Final softmax layer for binary classification

## 🔄 Training Pipeline

### Features
- Mixed precision training for improved performance
- Cosine annealing learning rate scheduler
- Weighted sampling for class balance
- Early stopping
- Model checkpointing
- Comprehensive augmentation pipeline

### Data Augmentation
- Random color jittering
- Random affine transformations
- Random horizontal/vertical flips
- Gaussian blur
- Random erasing
- Sharpness adjustment

## 📊 Evaluation Metrics

The testing pipeline provides:
- Accuracy, Sensitivity, and Specificity
- ROC curves with AUC scores
- Confusion matrices
- Detailed classification reports
- Per-dataset performance metrics

## 📝 Logging

The training process logs:
- Training and validation losses
- Learning rates
- Evaluation metrics
- Model parameters
- Timestamps

All logs are saved in JSON format for easy analysis.

## 🔍 Model Checkpointing

The trainer saves:
- Best model based on validation loss
- Latest model state
- Periodic checkpoints every 10 epochs
- Training history and parameters

## 📈 Visualization

The project includes visualization tools for:
- Training/validation loss curves
- ROC curves
- Confusion matrices
- Performance metrics across datasets

## ⚙️ Configuration

Key parameters that can be configured:
- Batch size
- Number of epochs
- Learning rate
- Weight decay
- Image size
- Model architecture
- Dropout rate
- Number of frozen blocks
- Early stopping patience
