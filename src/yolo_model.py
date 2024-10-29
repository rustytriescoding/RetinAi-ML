from ultralytics import YOLO

class DiscDetectionModel:
    def __init__(self, model_name='yolov8n.pt'):
        self.model = YOLO(model_name)
    
    def train(self, yaml_path, epochs=100, imgsz=640, batch=16):
        self.model.train(
            data=yaml_path,
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            patience=20,
            save=True,
            device=0  
        )
    
    def predict(self, image_path, conf=0.5):
        return self.model.predict(image_path, conf=conf)
