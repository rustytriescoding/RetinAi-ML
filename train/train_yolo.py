from sys import path
path.append('../src')
from yolo_dataset import DiscDatasetPrep
from yolo_model import DiscDetectionModel
from pathlib import Path

def main():
    root_dir = Path(__file__).parent.parent
    data_dir = root_dir / 'data' / 'dataset3'
    
    # Setup datasets
    train_dataset = DiscDatasetPrep(
        csv_path=data_dir / 'csvs' / 'yolo_train.csv'
    )
    
    val_dataset = DiscDatasetPrep(
        csv_path=data_dir / 'csvs' / 'yolo_val.csv'
    )
    
    
    yolo_dir = data_dir / 'yolo'
    yolo_dir.mkdir(exist_ok=True)
    
    # Prepare training data
    train_dataset.prepare_dataset(
        output_dir=yolo_dir / 'train',
        fundus_dir=data_dir / 'full-fundus',
        seg_dir=data_dir / 'optic-disc'
    )
    
    # Prepare validation data
    val_dataset.prepare_dataset(
        output_dir=yolo_dir / 'val',
        fundus_dir=data_dir / 'full-fundus',
        seg_dir=data_dir / 'optic-disc'
    )
    
    # Create YAML config
    train_dataset.create_yaml(
        yaml_path=yolo_dir / 'dataset.yaml',
        train_dir=str(yolo_dir / 'train' / 'images'),
        val_dir=str(yolo_dir / 'val' / 'images')
    )
    
    # Initialize and train model
    model = DiscDetectionModel(model_name='yolov8n.pt')  
    model.train(
        yaml_path=yolo_dir / 'dataset.yaml',
        epochs=100,
        imgsz=640,
        batch=16
    )

if __name__ == "__main__":
    main()
