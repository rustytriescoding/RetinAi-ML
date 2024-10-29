from pathlib import Path
from sys import path
path.append('../src')
from yolo_model import DiscDetectionModel
import cv2

class DiscVisualizer:
    def __init__(self, weights_path='runs/detect/train/weights/best.pt', conf_threshold=0.5):
        self.weights_path = Path(weights_path)
        self.conf_threshold = conf_threshold
        self.model = DiscDetectionModel(model_name=str(self.weights_path))
        
    def visualize(self, image_path, output_path=None):
        image_path = Path(image_path)
        
        if output_path is None:
            output_path = image_path.parent / f"{image_path.stem}_detected{image_path.suffix}"
        else:
            output_path = Path(output_path)
            
        results = self.model.predict(str(image_path), conf=self.conf_threshold)
        
        img = cv2.imread(str(image_path))
        if img is None:
            raise ValueError(f"Could not read image at {image_path}")
            
        detections = []
        
        # Draw boxes on the image
        for result in results:
            boxes = result.boxes
            for box in boxes:
                # Get coordinates and confidence
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                conf = float(box.conf[0])
                
                # Store detection
                detections.append((x1, y1, x2, y2, conf))
                
                # Draw rectangle
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Add confidence score
                label = f"ONH: {conf:.2f}"
                cv2.putText(img, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 
                           0.5, (0, 255, 0), 2)
        
        # Save the annotated image
        cv2.imwrite(str(output_path), img)
        
        return img, detections

def main():
    output_dir = Path('yolo_test_results')
    output_dir.mkdir(exist_ok=True)  
    
    visualizer = DiscVisualizer(
        weights_path='../runs/detect/train/weights/best.pt',
        conf_threshold=0.5
    )
    
    image_path = "../data/dataset1/raw/32_right.jpg"
    output_path = output_dir / "yolo_output.jpg"
    
    _, detections = visualizer.visualize(image_path, output_path)
    
    for i, (x1, y1, x2, y2, conf) in enumerate(detections):
        print(f"Detection {i+1}:")
        print(f"  Coordinates: ({x1}, {y1}) to ({x2}, {y2})")
        print(f"  Confidence: {conf:.2f}")

if __name__ == "__main__":
    main()
