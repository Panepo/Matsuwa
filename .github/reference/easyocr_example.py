#!/usr/bin/env python3
"""
Example usage of EasyOCR ONNX models for text detection and recognition.
"""

import onnxruntime as ort
import cv2
import numpy as np
from typing import List
import argparse
import os

class EasyOCR_ONNX:
    """ONNX implementation of EasyOCR for text detection and recognition."""
    
    def __init__(self, 
                 detector_path: str = "craft_mlt_25k_jpqd.onnx",
                 recognizer_path: str = "english_g2_jpqd.onnx"):
        """
        Initialize EasyOCR ONNX models.
        
        Args:
            detector_path: Path to CRAFT detection model
            recognizer_path: Path to text recognition model
        """
        print(f"Loading detector: {detector_path}")
        self.detector = ort.InferenceSession(detector_path)
        
        print(f"Loading recognizer: {recognizer_path}")
        self.recognizer = ort.InferenceSession(recognizer_path)
        
        # Character sets
        self.english_charset = '0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~ '
        self.latin_charset = self._get_latin_charset()
        
        # Determine charset based on model
        if "english" in recognizer_path.lower():
            self.charset = self.english_charset
        elif "latin" in recognizer_path.lower():
            self.charset = self.latin_charset
        else:
            self.charset = self.english_charset
    
    def _get_latin_charset(self) -> str:
        """Get extended Latin character set."""
        # This is a simplified version - in practice, you'd load the full 352-character set
        basic = '0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~ '
        extended = 'àáâãäåæçèéêëìíîïðñòóôõöøùúûüýþÿĀāĂăĄąĆćĈĉĊċČčĎďĐđĒēĔĕĖėĘęĚě'
        return basic + extended
    
    def preprocess_for_detection(self, image: np.ndarray, target_size: int = 640) -> np.ndarray:
        """Preprocess image for CRAFT text detection."""
        # Resize to target size
        image_resized = cv2.resize(image, (target_size, target_size))
        
        # Normalize to [0, 1]
        image_norm = image_resized.astype(np.float32) / 255.0
        
        # Convert HWC to CHW
        image_chw = np.transpose(image_norm, (2, 0, 1))
        
        # Add batch dimension
        image_batch = np.expand_dims(image_chw, axis=0)
        
        return image_batch
    
    def preprocess_for_recognition(self, text_region: np.ndarray) -> np.ndarray:
        """Preprocess text region for CRNN recognition."""
        # Convert to grayscale if needed
        if len(text_region.shape) == 3:
            gray = cv2.cvtColor(text_region, cv2.COLOR_RGB2GRAY)
        else:
            gray = text_region
        
        # Resize to model input size (32 height, 100 width)
        resized = cv2.resize(gray, (100, 32))
        
        # Normalize to [0, 1]
        normalized = resized.astype(np.float32) / 255.0
        
        # Add batch and channel dimensions [1, 1, 32, 100]
        input_batch = np.expand_dims(np.expand_dims(normalized, axis=0), axis=0)
        
        return input_batch
    
    def detect_text(self, image: np.ndarray) -> np.ndarray:
        """
        Detect text regions in image using CRAFT model.
        
        Args:
            image: Input image (RGB format)
            
        Returns:
            Detection output maps
        """
        # Preprocess
        input_batch = self.preprocess_for_detection(image)
        
        # Run inference
        outputs = self.detector.run(None, {"input": input_batch})
        
        # Ensure we return a numpy array
        if isinstance(outputs[0], np.ndarray):
            return outputs[0]
        else:
            return np.array(outputs[0])  # Convert to numpy array if needed
    
    def recognize_text(self, text_regions: List[np.ndarray]) -> List[str]:
        """
        Recognize text in detected regions.
        
        Args:
            text_regions: List of cropped text region images
            
        Returns:
            List of recognized text strings
        """
        results = []
        
        for region in text_regions:
            # Preprocess
            input_batch = self.preprocess_for_recognition(region)
            
            # Run inference
            outputs = self.recognizer.run(None, {"input": input_batch})
            
            # Ensure output is numpy array and decode text
            output_array = outputs[0] if isinstance(outputs[0], np.ndarray) else np.array(outputs[0])
            text = self._decode_text(output_array)
            results.append(text)
        
        return results
    
    def _decode_text(self, output: np.ndarray) -> str:
        """Decode recognition output to text string using greedy decoding."""
        # Get character indices with highest probability
        indices = np.argmax(output[0], axis=1)
        
        # Convert indices to characters
        text = ''
        prev_char = ''
        
        for idx in indices:
            if idx < len(self.charset) and idx > 0:  # Skip blank token (index 0)
                char = self.charset[idx]
                # Simple CTC-like decoding: skip repeated characters
                if char != prev_char:
                    text += char
                prev_char = char
        
        return text.strip()
    
    def extract_simple_regions(self, detection_output: np.ndarray, 
                             original_image: np.ndarray,
                             threshold: float = 0.3) -> List[np.ndarray]:
        """
        Extract text regions from detection output (simplified version).
        In practice, you'd implement proper CRAFT post-processing.
        """
        # This is a simplified implementation for demonstration
        # In practice, you'd use proper CRAFT post-processing to extract precise text regions
        
        h, w = original_image.shape[:2]
        
        # Handle different output shapes
        if len(detection_output.shape) == 4:  # [batch, channels, height, width]
            detection_map = detection_output[0, 0]  # First channel of first batch
        elif len(detection_output.shape) == 3:  # [channels, height, width]
            detection_map = detection_output[0]  # First channel
        else:
            detection_map = detection_output
        
        # Normalize detection map to [0, 1] if needed
        if detection_map.max() > 1.0:
            detection_map = detection_map / detection_map.max()
        
        # Lower threshold for better detection
        binary_map = (detection_map > threshold).astype(np.uint8) * 255
        binary_map = cv2.resize(binary_map, (w, h))
        
        # Apply morphological operations to improve detection
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        binary_map = cv2.morphologyEx(binary_map, cv2.MORPH_CLOSE, kernel)
        
        contours, _ = cv2.findContours(binary_map, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        text_regions = []
        for contour in contours:
            # Get bounding box
            x, y, w_box, h_box = cv2.boundingRect(contour)
            
            # Filter small regions but be more permissive
            if w_box > 15 and h_box > 8 and cv2.contourArea(contour) > 100:
                # Add some padding
                x = max(0, x - 2)
                y = max(0, y - 2)
                w_box = min(w - x, w_box + 4)
                h_box = min(h - y, h_box + 4)
                
                # Extract region from original image
                region = original_image[y:y+h_box, x:x+w_box]
                if region.size > 0:  # Make sure region is not empty
                    text_regions.append(region)
        
        # If no regions found with CRAFT, fall back to simple grid sampling
        if len(text_regions) == 0:
            print("  No CRAFT regions found, using fallback method...")
            # Sample some regions from the image for demonstration
            step_y, step_x = h // 4, w // 4
            for y in range(0, h - 32, step_y):
                for x in range(0, w - 100, step_x):
                    region = original_image[y:y+32, x:x+100]
                    if region.size > 0 and np.mean(region) < 240:  # Skip mostly white regions
                        text_regions.append(region)
                        if len(text_regions) >= 4:  # Limit to 4 samples
                            break
                if len(text_regions) >= 4:
                    break
        
        return text_regions


def main():
    parser = argparse.ArgumentParser(description="EasyOCR ONNX Example")
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument("--detector", type=str, default="craft_mlt_25k_jpqd.onnx", 
                       help="Path to detection model")
    parser.add_argument("--recognizer", type=str, default="english_g2_jpqd.onnx",
                       help="Path to recognition model")
    parser.add_argument("--output", type=str, help="Path to save output image with detections")
    
    args = parser.parse_args()
    
    # Check if files exist
    if not os.path.exists(args.image):
        print(f"Error: Image file not found: {args.image}")
        return
    
    if not os.path.exists(args.detector):
        print(f"Error: Detector model not found: {args.detector}")
        return
        
    if not os.path.exists(args.recognizer):
        print(f"Error: Recognizer model not found: {args.recognizer}")
        return
    
    # Initialize OCR
    print("Initializing EasyOCR ONNX...")
    ocr = EasyOCR_ONNX(args.detector, args.recognizer)
    
    # Load image
    print(f"Loading image: {args.image}")
    image = cv2.imread(args.image)
    if image is None:
        print(f"Error: Could not load image: {args.image}")
        return
    
    # Convert BGR to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Detect text
    print("Detecting text regions...")
    detection_output = ocr.detect_text(image_rgb)
    
    # Extract text regions (simplified)
    text_regions = ocr.extract_simple_regions(detection_output, image_rgb)
    print(f"Found {len(text_regions)} text regions")
    
    # Recognize text
    if text_regions:
        print("Recognizing text...")
        recognized_texts = ocr.recognize_text(text_regions)
        
        # Print results
        print(f"\nRecognized text ({len(recognized_texts)} regions):")
        print("-" * 50)
        for i, text in enumerate(recognized_texts):
            print(f"Region {i+1}: '{text}'")
    else:
        print("No text regions detected")
    
    # Save output image with bounding boxes (if requested)
    if args.output and text_regions:
        output_image = image.copy()
        # This would draw bounding boxes on the image
        cv2.imwrite(args.output, output_image)
        print(f"Output saved to: {args.output}")


if __name__ == "__main__":
    main()