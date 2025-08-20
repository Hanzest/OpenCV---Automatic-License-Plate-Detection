from ultralytics import YOLO
import cv2
import os
import numpy as np
from datetime import datetime

class LicensePlateCharacterReader:
    def __init__(self, model_path="yolo_training/LPR_model2/weights/best.pt"):
        """
        Initialize the license plate character reader
        
        Args:
            model_path: Path to the trained character recognition model
        """
        self.model_path = model_path
        self.input_dir = "detected_plates"
        self.output_dir = "read_detected_plates"
        
        # Create output directory
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            print(f"Created output directory: {self.output_dir}")
        
        # Load the trained character recognition model
        print(f"Loading character recognition model from: {model_path}")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at: {model_path}")
        
        self.model = YOLO(model_path)
        print("Character recognition model loaded successfully!")
        
        # Class names mapping (based on your dataset)
        self.class_names = [
            '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
            'A', 'B', 'C', 'CAR PLATE', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N',
            'No-Plate', 'No.Plate', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
            'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q',
            'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z'
        ]
    
    def read_characters_from_plate(self, image_path, confidence_threshold=0.5):
        """
        Read alphanumeric characters from a license plate image
        
        Args:
            image_path: Path to the license plate image
            confidence_threshold: Minimum confidence for character detection
            
        Returns:
            tuple: (detected_text, annotated_image, detection_details)
        """
        # Load the image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Could not load image: {image_path}")
            return None, None, None
        
        # Run character detection
        results = self.model(image, conf=confidence_threshold, verbose=False)
        
        detected_chars = []
        annotated_image = image.copy()
        
        for r in results:
            if r.boxes is not None:
                for box in r.boxes:
                    # Get box coordinates and confidence
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    confidence = box.conf[0].item()
                    class_id = int(box.cls[0].item())
                    
                    # Get character name
                    if 0 <= class_id < len(self.class_names):
                        char_name = self.class_names[class_id]
                        
                        # Filter out non-alphanumeric detections
                        if char_name not in ['CAR PLATE', 'No-Plate', 'No.Plate']:
                            detected_chars.append({
                                'char': char_name,
                                'confidence': confidence,
                                'bbox': (x1, y1, x2, y2),
                                'center_x': (x1 + x2) / 2,
                                'center_y': (y1 + y2) / 2
                            })
                            
                            # Draw bounding box and label
                            color = (0, 255, 0) if confidence > 0.7 else (0, 255, 255)
                            cv2.rectangle(annotated_image, (x1, y1), (x2, y2), color, 2)
                            
                            # Draw character label
                            label = f"{char_name}: {confidence:.2f}"
                            cv2.putText(annotated_image, label, (x1, y1 - 10),
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Sort characters by position (left to right, top to bottom)
        detected_chars.sort(key=lambda x: (x['center_y'], x['center_x']))
        
        # Extract text in reading order
        detected_text = ''.join([char['char'] for char in detected_chars])
        
        return detected_text, annotated_image, detected_chars
    
    def determine_plate_format(self, detected_chars, image_shape):
        """
        Determine if the plate is single line or double line format
        
        Args:
            detected_chars: List of detected character dictionaries
            image_shape: Shape of the image (height, width)
            
        Returns:
            str: 'single_line', 'double_line', or 'unknown'
        """
        if len(detected_chars) < 2:
            return 'unknown'
        
        height = image_shape[0]
        mid_height = height / 2
        
        # Separate characters into top and bottom halves
        top_chars = [char for char in detected_chars if char['center_y'] < mid_height]
        bottom_chars = [char for char in detected_chars if char['center_y'] >= mid_height]
        
        # If most characters are in one half, it's likely single line
        if len(top_chars) > len(bottom_chars) * 2 or len(bottom_chars) > len(top_chars) * 2:
            return 'single_line'
        elif len(top_chars) > 0 and len(bottom_chars) > 0:
            return 'double_line'
        else:
            return 'single_line'
    
    def format_detected_text(self, detected_chars, image_shape):
        """
        Format detected text based on plate layout
        
        Args:
            detected_chars: List of detected character dictionaries
            image_shape: Shape of the image
            
        Returns:
            str: Formatted text
        """
        if not detected_chars:
            return "No characters detected"
        
        plate_format = self.determine_plate_format(detected_chars, image_shape)
        
        if plate_format == 'double_line':
            height = image_shape[0]
            mid_height = height / 2
            
            # Separate and sort characters by line
            top_chars = sorted([char for char in detected_chars if char['center_y'] < mid_height],
                             key=lambda x: x['center_x'])
            bottom_chars = sorted([char for char in detected_chars if char['center_y'] >= mid_height],
                                key=lambda x: x['center_x'])
            
            top_text = ''.join([char['char'] for char in top_chars])
            bottom_text = ''.join([char['char'] for char in bottom_chars])
            
            if top_text and bottom_text:
                return f"{top_text}|{bottom_text}"
            elif top_text:
                return top_text
            else:
                return bottom_text
        else:
            # Single line format
            sorted_chars = sorted(detected_chars, key=lambda x: x['center_x'])
            return ''.join([char['char'] for char in sorted_chars])
    
    def process_all_plates(self, confidence_threshold=0.5):
        """
        Process all license plate images in the input directory
        
        Args:
            confidence_threshold: Minimum confidence for character detection
        """
        if not os.path.exists(self.input_dir):
            print(f"Input directory not found: {self.input_dir}")
            return
        
        # Get all image files
        image_files = [f for f in os.listdir(self.input_dir) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        if not image_files:
            print(f"No image files found in {self.input_dir}")
            return
        
        print(f"Found {len(image_files)} license plate images to process...")
        print(f"Using confidence threshold: {confidence_threshold}")
        print(f"Output directory: {self.output_dir}")
        print("="*60)
        
        results_summary = []
        
        for i, filename in enumerate(image_files, 1):
            print(f"Processing {i}/{len(image_files)}: {filename}")
            
            input_path = os.path.join(self.input_dir, filename)
            
            # Read characters from the plate
            detected_text, annotated_image, detected_chars = self.read_characters_from_plate(
                input_path, confidence_threshold
            )
            
            if detected_text is not None and annotated_image is not None:
                # Load original image for format detection
                original_image = cv2.imread(input_path)
                formatted_text = self.format_detected_text(detected_chars, original_image.shape)
                
                # Create output filename
                base_name = os.path.splitext(filename)[0]
                output_filename = f"{base_name}_read.jpg"
                output_path = os.path.join(self.output_dir, output_filename)
                
                # Add text overlay to the annotated image
                overlay_text = f"Detected: {formatted_text}"
                cv2.putText(annotated_image, overlay_text, (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                cv2.putText(annotated_image, overlay_text, (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 1)
                
                # Save annotated image
                cv2.imwrite(output_path, annotated_image)
                
                # Save text result
                text_filename = f"{base_name}_text.txt"
                text_path = os.path.join(self.output_dir, text_filename)
                with open(text_path, 'w') as f:
                    f.write(f"Original file: {filename}\\n")
                    f.write(f"Detected text: {formatted_text}\\n")
                    f.write(f"Character count: {len(detected_chars)}\\n")
                    f.write(f"Confidence threshold: {confidence_threshold}\\n")
                    f.write(f"Processing time: {datetime.now()}\\n\\n")
                    f.write("Character details:\\n")
                    for char_info in detected_chars:
                        f.write(f"  {char_info['char']}: {char_info['confidence']:.3f}\\n")
                
                results_summary.append({
                    'filename': filename,
                    'detected_text': formatted_text,
                    'char_count': len(detected_chars),
                    'output_image': output_filename
                })
                
                print(f"  ✅ Detected: '{formatted_text}' ({len(detected_chars)} characters)")
                print(f"  💾 Saved: {output_filename}")
            else:
                print(f"  ❌ Failed to process: {filename}")
            
            print("-" * 40)
        
        # Save summary report
        summary_path = os.path.join(self.output_dir, "processing_summary.txt")
        with open(summary_path, 'w') as f:
            f.write("LICENSE PLATE CHARACTER RECOGNITION SUMMARY\\n")
            f.write("=" * 50 + "\\n")
            f.write(f"Processing date: {datetime.now()}\\n")
            f.write(f"Model used: {self.model_path}\\n")
            f.write(f"Confidence threshold: {confidence_threshold}\\n")
            f.write(f"Total images processed: {len(image_files)}\\n")
            f.write(f"Successfully processed: {len(results_summary)}\\n\\n")
            
            for result in results_summary:
                f.write(f"File: {result['filename']}\\n")
                f.write(f"Text: {result['detected_text']}\\n")
                f.write(f"Characters: {result['char_count']}\\n")
                f.write(f"Output: {result['output_image']}\\n")
                f.write("-" * 30 + "\\n")
        
        print("="*60)
        print("PROCESSING COMPLETE!")
        print(f"✅ Successfully processed: {len(results_summary)}/{len(image_files)} images")
        print(f"📁 Results saved to: {self.output_dir}")
        print(f"📊 Summary report: {summary_path}")
        print("="*60)

def main():
    """Main function to run character recognition on detected plates"""
    try:
        # Initialize the character reader
        reader = LicensePlateCharacterReader(
            model_path="yolo_training/LPR_model2/weights/best.pt"
        )
        
        # Process all plates in the detected_plates directory
        reader.process_all_plates(confidence_threshold=0.5)
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please make sure the trained model exists at the specified path.")
    except Exception as e:
        print(f"Unexpected error: {e}")

if __name__ == "__main__":
    main()
