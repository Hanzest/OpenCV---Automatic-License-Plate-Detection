# Vietnamese License Plate Detection and Recognition System

A comprehensive YOLO-based system for detecting and reading Vietnamese license plates from images and live camera feeds. This project combines two deep learning models: one for plate detection and another for character recognition.

## 🚀 Features

- **Real-time License Plate Detection**: Uses YOLOv8 to detect license plates in live camera feeds
- **Character Recognition**: Reads and extracts text from detected license plates
- **Automatic Cropping**: Saves high-confidence detections for further analysis
- **GPU/CPU Support**: Optimized for both GPU and CPU inference
- **Batch Processing**: Process multiple images at once
- **Confidence Thresholding**: Configurable confidence levels for detection and cropping

## 📁 Project Structure

```
├── README.md                    # This file
├── TRAINING_SUMMARY.md         # Training progress and results
├── requirements.txt            # Python dependencies
│
├── Training Scripts/
│   ├── plate_train.py          # Train the license plate detection model
│   └── reader_train.py         # Train the character recognition model
│
├── Detection & Recognition/
│   ├── cropped_camera_detector.py  # Real-time camera detection with cropping
│   ├── gpu_detector.py             # GPU-optimized batch detection
│   ├── predict_plates.py           # Standalone plate prediction
│   └── reader_plates.py            # Character recognition from cropped plates
│
├── Models/
│   ├── yolo11n.pt              # Pre-trained YOLO11 base model
│   ├── yolov8n.pt              # Pre-trained YOLOv8 base model
│   └── yolo_training/          # Trained model weights
│       ├── vnlp_model7/        # Vietnamese license plate detection model
│       ├── LPR_model1/         # License plate reader model v1
│       └── LPR_model2/         # License plate reader model v2
│
├── Datasets/
│   ├── VNLP yolov8/           # Vietnamese license plate dataset
│   └── LP reader yolov8/       # Character recognition dataset
│
├── Output/
│   ├── detected_plates/        # Cropped license plates (high confidence)
│   ├── read_detected_plates/   # Processed plates with recognized text
│   ├── batch_predictions/      # Batch processing results
│   └── runs/                   # Training logs and metrics
│
└── Resources/
    ├── Photos/                 # Test images
    └── Videos/                 # Test videos
```

## 🔧 Installation

### Prerequisites
- Python 3.8+
- CUDA (optional, for GPU acceleration)

### Setup
1. Clone this repository:
```bash
git clone <repository-url>
cd Prototype4-Yolo
```

2. Install dependencies:
```bash
pip install ultralytics opencv-python torch torchvision numpy pillow
```

3. Ensure you have the trained models in the `yolo_training/` directory

## 🎯 Usage

### 1. Real-time License Plate Detection
Detect and crop license plates from live camera feed:
```bash
python cropped_camera_detector.py
```
- **ESC**: Exit the application
- **SPACE**: Pause/Resume detection
- Automatically saves high-confidence detections to `detected_plates/`

### 2. Character Recognition
Read text from detected license plates:
```bash
python reader_plates.py
```
- Processes all images in `detected_plates/`
- Outputs annotated images to `read_detected_plates/`

### 3. Batch Processing (GPU)
Process multiple images efficiently:
```bash
python gpu_detector.py
```

### 4. Single Image Prediction
Test the model on a single image:
```bash
python predict_plates.py
```

## 🏋️ Training

### Train License Plate Detection Model
```bash
python plate_train.py
```
- Uses Vietnamese license plate dataset (`VNLP yolov8/`)
- Trains YOLOv8 to detect license plate regions
- Saves model to `yolo_training/vnlp_model*/weights/best.pt`

### Train Character Recognition Model
```bash
python reader_train.py
```
- Uses character-level dataset (`LP reader yolov8/`)
- Trains YOLOv8 to recognize individual characters
- Saves model to `yolo_training/LPR_model*/weights/best.pt`

## ⚙️ Configuration

### Key Parameters in `cropped_camera_detector.py`:
- `confidence_threshold`: Minimum confidence for displaying detections (default: 0.5)
- `crop_threshold`: Minimum confidence for saving crops (default: 0.9)
- `plate_model_path`: Path to plate detection model
- `char_model_path`: Path to character recognition model

### Model Paths:
- **Plate Detection**: `yolo_training/vnlp_model7/weights/best.pt`
- **Character Recognition**: `yolo_training/LPR_model2/weights/best.pt`

## 📊 Performance

### Detection Model (vnlp_model7)
- **Architecture**: YOLOv8n
- **Dataset**: 1,005 Vietnamese license plate images
- **Classes**: 1 (license plate)

### Character Recognition Model (LPR_model2)
- **Architecture**: YOLOv8n
- **Classes**: Alphanumeric characters (A-Z, 0-9)
- **Purpose**: Extract text from cropped license plates

## 🔍 Pipeline Workflow

1. **Input**: Live camera feed or image
2. **Detection**: Plate detection model identifies license plate regions
3. **Cropping**: High-confidence detections are cropped and saved
4. **Recognition**: Character recognition model reads text from cropped plates
5. **Output**: Annotated frames with license plate text

## 📋 Requirements

```txt
ultralytics>=8.0.0
opencv-python>=4.5.0
torch>=1.9.0
torchvision>=0.10.0
numpy>=1.21.0
pillow>=8.0.0
```

## 🐛 Troubleshooting

### Common Issues:

1. **Model not found**: Ensure trained models exist in `yolo_training/` directory
2. **CUDA errors**: Install CUDA-compatible PyTorch version
3. **Camera not detected**: Check camera permissions and connections
4. **Low accuracy**: Retrain models with more diverse datasets

### Performance Optimization:

- **GPU**: Use `gpu_detector.py` for faster batch processing
- **CPU**: Reduce batch size for memory constraints
- **Confidence**: Adjust thresholds based on accuracy requirements

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) for the base detection framework
- Vietnamese license plate dataset contributors
- OpenCV community for computer vision tools

## 📞 Contact

For questions or support, please open an issue in this repository.

---

**Note**: This system is designed for educational and research purposes. Ensure compliance with local privacy laws when using with live camera feeds.
