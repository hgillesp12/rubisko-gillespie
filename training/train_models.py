from ultralytics import YOLO

def train_model(task_name, data_yaml):
    # Load a base model
    model = YOLO('yolov8n.pt') 
    
    # Train
    model.train(
        data=data_yaml,
        epochs=50,
        imgsz=640,
        name=task_name
    )

if __name__ == "__main__":
    train_model('contamination', 'training/contamination_data.yaml')
    train_model('quality', 'training/quality_data.yaml')
    train_model('stage', 'training/stage_data.yaml')
    train_model('ratios', 'training/ratios_data.yaml')