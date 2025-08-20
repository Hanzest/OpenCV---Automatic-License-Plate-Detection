import cv2
import numpy as np
from ultralytics import YOLO
import os
from datetime import datetime

def preprocess_image(image):
    # Convert image to grayscale
    # Stretch the image to double its width
    # Blurred the image to reduce noise
    gray = image.convert('L')  # Convert to grayscale

    width, height = gray.size
    new_width = int(width * 2)
    new_height = height
    stretched = gray.resize((new_width, new_height))
    return stretched

class CameraLicensePlateDetector:
    def __init__(self, plate_model_path="yolo_training/vnlp_model7/weights/best.pt", 
                 char_model_path="yolo_training/LPR_model2/weights/best.pt",
                 confidence_threshold=0.5, crop_threshold=0.9):
        """
        Initialize the camera license plate detector with character recognition
        
        Args:
            plate_model_path: Path to the trained plate detection YOLO model
            char_model_path: Path to the trained character recognition YOLO model
            confidence_threshold: Minimum confidence for showing detections
            crop_threshold: Minimum confidence for cropping detected plates
        """
        self.plate_model_path = plate_model_path
        self.char_model_path = char_model_path
        self.confidence_threshold = confidence_threshold
        self.crop_threshold = crop_threshold
        self.crop_counter = 0
        
        # Create directory for saved crops
        self.crop_dir = "detected_plates"
        if not os.path.exists(self.crop_dir):
            os.makedirs(self.crop_dir)
        
        # Load the trained plate detection model
        print(f"Loading plate detection model from: {plate_model_path}")
        self.plate_model = YOLO(plate_model_path)
        print("Plate detection model loaded successfully!")
        
        # Load the trained character recognition model
        print(f"Loading character recognition model from: {char_model_path}")
        if not os.path.exists(char_model_path):
            print(f"Warning: Character model not found at {char_model_path}")
            print("Character recognition will be disabled.")
            self.char_model = None
        else:
            self.char_model = YOLO(char_model_path)
            print("Character recognition model loaded successfully!")
        
        # Class names mapping for character recognition (based on your dataset)
        self.char_class_names = [
            '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
            'A', 'B', 'C', 'CAR PLATE', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N',
            'No-Plate', 'No.Plate', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
            'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q',
            'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z'
        ]
        
        # Initialize camera
        self.cap = None
        
    def start_camera(self, camera_id=0):
        """Start the camera capture"""
        self.cap = cv2.VideoCapture(camera_id)
        if not self.cap.isOpened():
            raise Exception(f"Could not open camera {camera_id}")
        
        # Set camera properties for better performance
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        
    def crop_plate(self, frame, box, confidence):
        """
        Crop the detected license plate from the frame
        
        Args:
            frame: Original frame
            box: Detection bounding box
            confidence: Detection confidence
        """
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        
        # Add some padding around the detected plate
        padding = 10
        h, w = frame.shape[:2]
        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = min(w, x2 + padding)
        y2 = min(h, y2 + padding)
        
        # Crop the plate
        cropped_plate = frame[y1:y2, x1:x2]
        
        # Save the cropped plate
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"plate_{timestamp}_{self.crop_counter:04d}_conf{confidence:.3f}.jpg"
        filepath = os.path.join(self.crop_dir, filename)
        
        cv2.imwrite(filepath, cropped_plate)
        self.crop_counter += 1
        
        print(f"High confidence plate detected! Saved: {filename} (conf: {confidence:.3f})")
        return cropped_plate, filepath
    
    def read_characters_from_plate(self, cropped_plate, confidence_threshold=0.5):
        """
        Read alphanumeric characters from a cropped license plate image
        
        Args:
            cropped_plate: Cropped license plate image (numpy array)
            confidence_threshold: Minimum confidence for character detection
            
        Returns:
            str: Detected text from the license plate
        """
        if self.char_model is None:
            return "No char model"
        
        if cropped_plate is None or cropped_plate.size == 0:
            return "No plate"
        
        try:
            # Run character detection on the cropped plate
            results = self.char_model(cropped_plate, conf=confidence_threshold, verbose=False)
            
            detected_chars = []
            
            for r in results:
                if r.boxes is not None:
                    for box in r.boxes:
                        # Get box coordinates and confidence
                        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                        confidence = box.conf[0].item()
                        class_id = int(box.cls[0].item())
                        
                        # Get character name
                        if 0 <= class_id < len(self.char_class_names):
                            char_name = self.char_class_names[class_id]
                            
                            # Filter out non-alphanumeric detections
                            if char_name not in ['CAR PLATE', 'No-Plate', 'No.Plate']:
                                detected_chars.append({
                                    'char': char_name,
                                    'confidence': confidence,
                                    'center_x': (x1 + x2) / 2,
                                    'center_y': (y1 + y2) / 2
                                })
            
            if not detected_chars:
                return "No chars"
            
            # Determine plate format and sort characters
            formatted_text = self.format_detected_text(detected_chars, cropped_plate.shape)
            return formatted_text
            
        except Exception as e:
            return f"Error: {str(e)[:10]}"
    
    def format_detected_text(self, detected_chars, image_shape):
        """
        Format detected text based on plate layout
        
        Args:
            detected_chars: List of detected character dictionaries
            image_shape: Shape of the cropped plate image
            
        Returns:
            str: Formatted text
        """
        if not detected_chars:
            return "No chars"
        
        height = image_shape[0]
        mid_height = height / 2
        
        # Separate characters into top and bottom halves
        top_chars = [char for char in detected_chars if char['center_y'] < mid_height]
        bottom_chars = [char for char in detected_chars if char['center_y'] >= mid_height]
        
        # Determine if it's single line or double line
        if len(top_chars) > 0 and len(bottom_chars) > 0 and len(top_chars) + len(bottom_chars) > 3:
            # Double line format
            top_chars.sort(key=lambda x: x['center_x'])
            bottom_chars.sort(key=lambda x: x['center_x'])
            
            top_text = ''.join([char['char'] for char in top_chars])
            bottom_text = ''.join([char['char'] for char in bottom_chars])
            
            if top_text and bottom_text:
                return f"{top_text}|{bottom_text}"
            elif top_text:
                return top_text
            else:
                return bottom_text
        else:
            # Single line format - sort all characters by x position
            all_chars = sorted(detected_chars, key=lambda x: x['center_x'])
            return ''.join([char['char'] for char in all_chars])
    
    def draw_detections(self, frame, results):
        """
        Draw bounding boxes and labels on the frame with character recognition
        
        Args:
            frame: Original frame
            results: YOLO detection results from plate detection model
            
        Returns:
            frame: Frame with drawn detections and recognized text
        """
        cropped_plates = []
        
        for r in results:
            if r.boxes is not None:
                for box in r.boxes:
                    # Get box coordinates and confidence
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    confidence = box.conf[0].item()
                    
                    # Only show detections above threshold
                    if confidence >= self.confidence_threshold:
                        # Choose color based on confidence
                        if confidence >= self.crop_threshold:
                            color = (0, 255, 0)  # Green for high confidence
                            thickness = 3
                            
                            # Crop the plate if confidence is high enough
                            cropped_plate, filepath = self.crop_plate(frame, box, confidence)
                            cropped_plates.append((cropped_plate, filepath, confidence))
                            
                            # Read characters from the cropped plate
                            detected_text = self.read_characters_from_plate(cropped_plate, confidence_threshold=0.4)
                            
                        else:
                            color = (0, 255, 255)  # Yellow for medium confidence
                            thickness = 2
                            
                            # For medium confidence, still try to read characters but don't save
                            # Create a temporary crop for character recognition
                            temp_x1, temp_y1, temp_x2, temp_y2 = x1, y1, x2, y2
                            padding = 10
                            h, w = frame.shape[:2]
                            temp_x1 = max(0, temp_x1 - padding)
                            temp_y1 = max(0, temp_y1 - padding)
                            temp_x2 = min(w, temp_x2 + padding)
                            temp_y2 = min(h, temp_y2 + padding)
                            temp_crop = frame[temp_y1:temp_y2, temp_x1:temp_x2]
                            
                            detected_text = self.read_characters_from_plate(temp_crop, confidence_threshold=0.4)
                        
                        # Draw bounding box
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
                        
                        # Draw confidence label at top-left of bounding box
                        conf_label = f"Plate: {confidence:.2f}"
                        conf_label_size, _ = cv2.getTextSize(conf_label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                        
                        # Calculate label position and ensure it doesn't go outside frame
                        label_height = conf_label_size[1] + 10
                        label_y_top = max(0, y1 - label_height)  # Ensure it doesn't go above frame
                        label_y_bottom = label_y_top + label_height
                        
                        # If the label would go outside the top, place it inside the bounding box
                        if label_y_top == 0:
                            label_y_top = y1
                            label_y_bottom = y1 + label_height
                        
                        cv2.rectangle(frame, (x1, label_y_top + 15), 
                                    (x1 + conf_label_size[0], label_y_bottom + 15), color, -1)
                        cv2.putText(frame, conf_label, (x1, label_y_bottom + 10), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
                        
                        # Draw detected text at bottom-right corner of bounding box
                        if detected_text and detected_text not in ["No chars", "No plate", "No char model"]:
                            text_label = f"{detected_text}"
                            text_label_size, _ = cv2.getTextSize(text_label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                            
                            # Position text at bottom-right of bounding box
                            text_x = x2 - text_label_size[0] - 5
                            text_y = y2 + text_label_size[1] - 10
                            
                            # Ensure text doesn't go out of frame
                            text_x = max(0, text_x)
                            text_y = min(frame.shape[0] - 5, text_y)
                            
                            # Draw background rectangle for text
                            cv2.rectangle(frame, (text_x - 5, text_y - text_label_size[1] - 5), 
                                        (text_x + text_label_size[0] + 5, text_y + 5), (0, 0, 0), -1)
                            
                            # Draw the detected text
                            cv2.putText(frame, text_label, (text_x, text_y), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                        else:
                            # Draw "Reading..." if no text detected
                            reading_label = "Reading..."
                            reading_label_size, _ = cv2.getTextSize(reading_label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                            reading_x = x2 - reading_label_size[0]
                            reading_y = y2 + reading_label_size[1] + 5
                            reading_x = max(0, reading_x)
                            reading_y = min(frame.shape[0] - 5, reading_y)
                            
                            cv2.rectangle(frame, (reading_x - 3, reading_y - reading_label_size[1] - 3), 
                                        (reading_x + reading_label_size[0] + 3, reading_y + 3), (50, 50, 50), -1)
                            cv2.putText(frame, reading_label, (reading_x, reading_y), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        return frame, cropped_plates
    
    def run_detection(self):
        """
        Main detection loop
        """
        if self.cap is None:
            self.start_camera()
        
        print("Starting license plate detection...")
        print(f"Detection threshold: {self.confidence_threshold}")
        print(f"Crop threshold: {self.crop_threshold}")
        print("Press 'q' to quit, 's' to save current frame, 'c' to clear crop counter")
        
        frame_count = 0
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("Failed to grab frame")
                break
            
            frame_count += 1
            
            # Run detection every frame (you can skip frames for better performance)
            if frame_count % 1 == 0:  # Process every frame
                results = self.plate_model(frame, conf=self.confidence_threshold, verbose=False)
                
                # Draw detections with character recognition
                frame, cropped_plates = self.draw_detections(frame, results)
                
                # Show any newly cropped plates in separate windows
                for i, (cropped_plate, filepath, conf) in enumerate(cropped_plates):
                    if cropped_plate.shape[0] > 20 and cropped_plate.shape[1] > 20:  # Only show if big enough
                        # Resize cropped plate for better visibility
                        height, width = cropped_plate.shape[:2]
                        if width < 200:
                            scale = 200 / width
                            new_width = int(width * scale)
                            new_height = int(height * scale)
                            cropped_plate = cv2.resize(cropped_plate, (new_width, new_height))
                        
                        cv2.imshow(f"Detected Plate - {conf:.3f}", cropped_plate)
            
            # Add frame info
            info_text = f"Frame: {frame_count} | Crops saved: {self.crop_counter}"
            cv2.putText(frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Add threshold info
            threshold_text = f"Detection: {self.confidence_threshold} | Crop: {self.crop_threshold}"
            cv2.putText(frame, threshold_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Show the frame
            cv2.imshow('License Plate Detection', frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                # Save current frame
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"frame_{timestamp}.jpg"
                cv2.imwrite(filename, frame)
                print(f"Frame saved as: {filename}")
            elif key == ord('c'):
                # Clear crop counter
                self.crop_counter = 0
                print("Crop counter reset")
            elif key == ord('+') or key == ord('='):
                # Increase crop threshold
                self.crop_threshold = min(1.0, self.crop_threshold + 0.05)
                print(f"Crop threshold: {self.crop_threshold:.2f}")
            elif key == ord('-'):
                # Decrease crop threshold
                self.crop_threshold = max(0.1, self.crop_threshold - 0.05)
                print(f"Crop threshold: {self.crop_threshold:.2f}")
    
    def cleanup(self):
        """Clean up resources"""
        if self.cap is not None:
            self.cap.release()
        cv2.destroyAllWindows()
        print(f"Detection complete. {self.crop_counter} plates saved to '{self.crop_dir}' folder.")

def main():
    """Main function to run the camera detector"""
    detector = None
    try:
        # Create detector instance
        detector = CameraLicensePlateDetector(
            plate_model_path="yolo_training/vnlp_model7/weights/best.pt",
            char_model_path="yolo_training/LPR_model2/weights/best.pt",
            confidence_threshold=0.3,  # Show detections above 30%
            crop_threshold=0.88         # Only crop/save plates above 88%
        )
        
        # Start detection
        detector.run_detection()
        
    except KeyboardInterrupt:
        print("\nDetection interrupted by user")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        if detector is not None:
            detector.cleanup()

if __name__ == "__main__":
    main()  
 