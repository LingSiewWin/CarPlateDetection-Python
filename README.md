# Malaysian License Plate Detection System

This system is designed to detect and recognize Malaysian license plates from images using traditional image processing techniques. The system uses OpenCV for image processing and Tesseract OCR for text recognition.

## Prerequisites

1. Python 3.7 or higher
2. Tesseract OCR installed on your system
3. Required Python packages (listed in requirements.txt)

## Installation

1. Install Tesseract OCR:
   - For macOS: `brew install tesseract`
   - For Ubuntu: `sudo apt-get install tesseract-ocr`
   - For Windows: Download and install from https://github.com/UB-Mannheim/tesseract/wiki

2. Install required Python packages:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Run the program:
   ```bash
   python license_plate_detector.py
   ```

2. Use the GUI to:
   - Click "Select Image" to choose an image containing a vehicle
   - Click "Detect License Plate" to process the image
   - View the results in the GUI

## Features

- Image preprocessing for better plate detection
- License plate detection using contour analysis
- Text extraction using OCR
- Simple GUI interface
- Support for various image formats

## Notes

- The system works best with clear, well-lit images
- The license plate should be clearly visible in the image
- The system is optimized for Malaysian license plate formats

## Limitations

- Performance may vary based on image quality and lighting conditions
- May not work well with heavily distorted or damaged plates
- Processing speed depends on image size and system specifications 