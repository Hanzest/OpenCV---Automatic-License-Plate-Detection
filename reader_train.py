from ultralytics import YOLO
import os
import torch
import time
import glob

def find_next_model_index():
    """Find the next available model index"""
    pattern = "yolo_training/LPR_model*"
    existing_dirs = glob.glob(pattern)
    if not existing_dirs:
        return 1
    
    # Extract indices from existing directories
    indices = []
    for dir_path in existing_dirs:
        try:
            index = int(dir_path.split("LPR_model")[-1])
            indices.append(index)
        except ValueError:
            continue
    
    return max(indices) + 1 if indices else 1

def validate_dataset_structure():
    """Validate the dataset structure"""
    base_path = "LP reader yolov8"
    data_yaml_path = os.path.join(base_path, "data.yaml")
    
    print("Validating dataset structure...")
    
    if not os.path.exists(data_yaml_path):
        raise FileNotFoundError(f"data.yaml not found at: {data_yaml_path}")
    
    # Check train, val, test directories
    for split in ['train', 'valid', 'test']:
        images_dir = os.path.join(base_path, split, 'images')
        labels_dir = os.path.join(base_path, split, 'labels')
        
        if not os.path.exists(images_dir):
            print(f"Warning: {images_dir} not found")
        else:
            image_count = len([f for f in os.listdir(images_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
            print(f"  {split}/images: {image_count} images")
        
        if not os.path.exists(labels_dir):
            print(f"Warning: {labels_dir} not found")
        else:
            label_count = len([f for f in os.listdir(labels_dir) if f.lower().endswith('.txt')])
            print(f"  {split}/labels: {label_count} labels")
    
    print("Dataset validation complete!")
    return data_yaml_path

def main():
    """Main training function for License Plate Character Recognition"""
    print("=" * 60)
    print("LICENSE PLATE CHARACTER RECOGNITION TRAINING")
    print("=" * 60)
    
    print(f"Python version: {torch.__version__ if hasattr(torch, '__version__') else 'Unknown'}")
    print(f"PyTorch version: {torch.__version__}")
    
    # Validate dataset structure
    try:
        data_yaml_path = validate_dataset_structure()
    except Exception as e:
        print(f"Dataset validation failed: {e}")
        return
    
    # Find next available model index
    model_index = find_next_model_index()
    model_name = f"LPR_model{model_index}"
    print(f"Training model: {model_name}")
    
    # Load a pretrained YOLOv8 model (NOT the VNLP model)
    print("Loading pretrained YOLOv8 model...")
    model = YOLO("yolov8n.pt")  # Use standard pretrained YOLOv8 nano
    print("Model loaded successfully!")
    
    # Check CUDA availability and set device
    if torch.cuda.is_available():
        device = 0  # Use GPU
        print(f"CUDA available! Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA version: {torch.version.cuda}")
        batch_size = 16  # Smaller batch for character recognition
        epochs = 100
        patience = 15
    else:
        device = 'cpu'  # Use CPU
        print("CUDA not available. Using CPU for training.")
        print("Note: Training will be slower on CPU. Consider installing CUDA-enabled PyTorch.")
        batch_size = 8  # Smaller batch for CPU
        epochs = 5  # Very short training for testing
        patience = 3
    
    # Create training directory
    training_dir = f"yolo_training/{model_name}"
    os.makedirs("yolo_training", exist_ok=True)
    
    print("\\nTraining Configuration:")
    print(f"  Dataset: {data_yaml_path}")
    print(f"  Model: YOLOv8n (pretrained)")
    print(f"  Epochs: {epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  Image size: 640")
    print(f"  Device: {device}")
    print(f"  Output directory: {training_dir}")
    print(f"  Classes: 65 (0-9, A-Z, a-z, special)")
    
    # Train the model for License Plate Character Recognition
    print("\\nStarting training...")
    start_time = time.time()
    
    try:
        results = model.train(
            data=data_yaml_path,  # Path to LP reader dataset
            epochs=epochs,        # More epochs for character recognition
            imgsz=640,           # Image size
            batch=batch_size,    # Adjust batch size based on device
            patience=patience,   # Early stopping patience
            save=True,           # Save model checkpoints
            project="yolo_training",  # Project name
            name=model_name,     # Run name with index
            device=device,       # Use detected device
            workers=0,           # Fix multiprocessing issue on Windows
            verbose=True,        # Verbose output
            val=True,            # Validate during training
            plots=True,          # Generate training plots
            save_period=10       # Save checkpoint every 10 epochs
        )
        
        training_time = time.time() - start_time
        print(f"\\nTraining completed successfully in {training_time/60:.1f} minutes")
        
        # Path to the best trained model
        best_model_path = f"yolo_training/{model_name}/weights/best.pt"
        last_model_path = f"yolo_training/{model_name}/weights/last.pt"
        
        print(f"\\nModel files saved:")
        print(f"  Best model: {best_model_path}")
        print(f"  Last model: {last_model_path}")
        
        # Load and test the best trained model
        if os.path.exists(best_model_path):
            print("\\nLoading best trained model for validation...")
            trained_model = YOLO(best_model_path)
            
            # Get model info
            model_info = trained_model.info()
            print(f"Model summary: {model_info}")
            
            # Run validation
            print("Running final validation...")
            val_results = trained_model.val()
            print(f"Final validation mAP50: {val_results.box.map50:.4f}")
            print(f"Final validation mAP50-95: {val_results.box.map:.4f}")
        
        print(f"\\n{'='*60}")
        print("TRAINING SUMMARY")
        print(f"{'='*60}")
        print(f"Model: {model_name}")
        print(f"Training time: {training_time/60:.1f} minutes")
        print(f"Best model: {best_model_path}")
        print(f"Dataset: License Plate Character Recognition (65 classes)")
        print(f"Use this model for extracting alphanumeric characters from license plates!")
        print(f"{'='*60}")
        
    except Exception as e:
        print(f"\\nTraining failed with error: {e}")
        print("Please check your dataset and try again.")
        raise

if __name__ == "__main__":
    main()
