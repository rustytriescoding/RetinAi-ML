# Glaucoma Detection Project

This repository contains code used to train and test a glaucoma diagnosis model using a Resnet, EfficientNet, and MobileNetv3. Also contains YOLOv8 code to train and test a model to locate the ONH (optic nerve head). There are scripts to create training, validation, and testing csvs, obtain dataset statistics, and crop the optic nerve head using the YOLOv8 model.

## Project Structure

```
├── .vscode/                  # VSCode configuration
├── data/
│   ├── dataset1/
│   │   ├── csvs/             # csv files (dataset, test, train, validation)
│   │   ├── disc-crop/        # Cropped fundus images
│   │   └── raw/              # Original fundus images
│   ├── dataset2/
│   │   ├── csvs/
│   │   ├── disc-crop/
│   │   └── raw/
│   ├── dataset3/
│   │   ├── csvs/            
│   │   ├── disc-crop/       
│   │   └── raw/             
│   ├── dataset4/
│   │   ├── csvs/
│   │   ├── disc-crop/
│   │   └── raw/
│   └── scripts/             # Data preprocessing and formatting scripts
├── models/                  # Saved models
├── src/
│   ├── pycache/
│   ├── glaucoma_dataset.py  # Glaucoma dataset
│   ├── glaucoma_model.py    # Glaucoma custom model 
│   ├── yolo_dataset.py      # YOLOv8 dataset
│   └── yolo_model.py        # YOLOv8 ONH detection model
├── tests/
│   ├── test_results/        # Test output and metrics
│   ├── test_model.py        # Test glaucoma diagnosis model
│   └── test_yolo.py         # Test YOLOv8 ONH detection model
├── train/
│   ├── model_checkpoints/   # Training checkpoints
│   ├── checkpoint_model.py  # Checkpoint class
│   ├── train_model.py       # Train glaucoma diagnosis model
│   └── train_yolo.py        # Train YOLOv8 ONH detection model
├── venv/                    # Python virtual environment
├── .gitignore
├── README.md
└── requirements.txt         # Project dependencies
```

## Requirements

- Python 3.12

## Getting Started

1. **Setup Environment**
   ```bash
   python -m venv venv
   source venv/bin/activate 
   pip install -r requirements.txt
   ```

2. **Prepare Dataset**
   - Obtain dataset folders from RetinAi Onedrive folder and place into data subfolder

3. **Training**
   - Configure 
   ```bash
   cd train
   py train_model.py
   ```

4. **Testing**
   ```bash
   cd tests
   python test_model.py
   ```
