import cv2
import numpy as np

def detect_license_plate(image):
    """
    Detect license plate in an image using enhanced edge detection and contour analysis.
    Optimized for both cars and buses with Malaysian license plates.
    
    Args:
        image: Input image containing vehicle with license plate
        
    Returns:
        Cropped image of the license plate region and coordinates
    """
    # Create a copy of the original image
    result = image.copy()
    
    # Resize image if it's too large (helps with processing speed)
    height, width = image.shape[:2]
    max_dimension = 1200
    if max(height, width) > max_dimension:
        scale = max_dimension / max(height, width)
        image = cv2.resize(image, None, fx=scale, fy=scale)
        height, width = image.shape[:2]
    
    # Initialize license plate contour
    license_plate = None
    x, y, w, h = 0, 0, 0, 0
    
    # FIRST APPROACH: Try color-based detection for black plates (common in Malaysian cars)
    # Convert to HSV color space
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Define range for black color - more precise for Malaysian plates
    lower_black = np.array([0, 0, 0])
    upper_black = np.array([180, 100, 60])  # Slightly increased value range for better detection
    
    # Create a mask for black regions
    mask_black = cv2.inRange(hsv, lower_black, upper_black)
    
    # Apply morphological operations to clean up the mask
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    mask_black = cv2.morphologyEx(mask_black, cv2.MORPH_CLOSE, kernel)
    
    # Find contours in the black mask
    contours_black, _ = cv2.findContours(mask_black, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Sort contours by area
    contours_black = sorted(contours_black, key=cv2.contourArea, reverse=True)[:15]
    
    # Look for rectangular black regions with appropriate aspect ratio
    for contour in contours_black:
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = w / float(h)
        area = w * h
        
        # Malaysian plates typically have aspect ratio between 2.0 and 5.5
        # Buses might have slightly different aspect ratios
        if 1.8 <= aspect_ratio <= 6.0 and area > 1000 and area < (width * height * 0.15):
            # Check if the region is in a reasonable position for a license plate
            if 0.2 <= (y + h/2) / height <= 0.95:  # Extended range for buses
                # Extract the license plate
                license_plate = image[y:y+h, x:x+w]
                return license_plate, (x, y, w, h)
    
    # SECOND APPROACH: Try white plates (common in some Malaysian vehicles)
    # Define range for white color
    lower_white = np.array([0, 0, 170])
    upper_white = np.array([180, 40, 255])
    
    # Create a mask for white regions
    mask_white = cv2.inRange(hsv, lower_white, upper_white)
    
    # Apply morphological operations
    mask_white = cv2.morphologyEx(mask_white, cv2.MORPH_CLOSE, kernel)
    
    # Find contours in the white mask
    contours_white, _ = cv2.findContours(mask_white, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Sort contours by area
    contours_white = sorted(contours_white, key=cv2.contourArea, reverse=True)[:15]
    
    for contour in contours_white:
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = w / float(h)
        area = w * h
        
        # More flexible aspect ratio for different plate types
        if 1.8 <= aspect_ratio <= 6.0 and area > 1000 and area < (width * height * 0.15):
            if 0.2 <= (y + h/2) / height <= 0.95:
                license_plate = image[y:y+h, x:x+w]
                return license_plate, (x, y, w, h)
    
    # THIRD APPROACH: Try edge detection (works well for both car and bus plates)
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply bilateral filter to reduce noise while preserving edges
    blur = cv2.bilateralFilter(gray, 11, 17, 17)
    
    # Apply Canny edge detection with improved parameters
    edges = cv2.Canny(blur, 30, 200)
    
    # Perform morphological operations to close gaps in edges
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    
    # Find contours
    contours, _ = cv2.findContours(edges.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    # Sort contours by area, largest first
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:20]  # Increased to 20 for buses
    
    # Loop through contours to find the license plate
    for contour in contours:
        # Approximate the contour
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
        
        # If the contour has 4-8 vertices, it's potentially a license plate
        # Buses might have more complex plate shapes
        if 4 <= len(approx) <= 8:
            x, y, w, h = cv2.boundingRect(contour)
            
            # Check aspect ratio and minimum area
            aspect_ratio = w / float(h)
            area = w * h
            
            # More flexible aspect ratio range for different vehicle types
            if 1.8 <= aspect_ratio <= 6.0 and area > 1000 and area < (width * height * 0.15):
                # Check position - buses might have plates in different positions
                if 0.2 <= (y + h/2) / height <= 0.95:
                    license_plate = image[y:y+h, x:x+w]
                    return license_plate, (x, y, w, h)
    
    # FOURTH APPROACH: Try with the regular bounding rectangle approach
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = w / float(h)
        area = w * h
        
        # More flexible parameters for buses and cars
        if 1.8 <= aspect_ratio <= 6.0 and area > 1000 and area < (width * height * 0.15):
            if 0.2 <= (y + h/2) / height <= 0.95:
                license_plate = image[y:y+h, x:x+w]
                return license_plate, (x, y, w, h)
    
    # FIFTH APPROACH: Special case for bus front plates (often larger and positioned differently)
    # Apply adaptive thresholding
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                  cv2.THRESH_BINARY_INV, 19, 9)
    
    # Find contours in threshold image
    contours_thresh, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    # Sort by area
    contours_thresh = sorted(contours_thresh, key=cv2.contourArea, reverse=True)[:20]
    
    for contour in contours_thresh:
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = w / float(h)
        area = w * h
        
        # Bus plates can sometimes be larger and have different aspect ratios
        if 1.5 <= aspect_ratio <= 7.0 and area > 1500 and area < (width * height * 0.2):
            # Bus plates can be higher up on the vehicle
            if 0.1 <= (y + h/2) / height <= 0.95:
                license_plate = image[y:y+h, x:x+w]
                return license_plate, (x, y, w, h)
    
    # Return the detected license plate and its coordinates
    # If no plate was found, this will return None and (0,0,0,0)
    return license_plate, (x, y, w, h)

# Add this code to make the script runnable from the terminal
if __name__ == "__main__":
    import sys
    import os
    
    # Check if an image path was provided as a command-line argument
    if len(sys.argv) < 2:
        print("Usage: python plate_detection.py <path_to_image>")
        sys.exit(1)
    
    # Get the image path from command-line arguments
    image_path = sys.argv[1]
    
    # Check if the file exists
    if not os.path.exists(image_path):
        print(f"Error: Image file '{image_path}' not found.")
        sys.exit(1)
    
    # Read the image
    image = cv2.imread(image_path)
    
    if image is None:
        print(f"Error: Could not read image '{image_path}'.")
        sys.exit(1)
    
    # Detect license plate
    plate, coords = detect_license_plate(image)
    
    if plate is None:
        print("No license plate detected in the image.")
        sys.exit(0)
    
    # Draw rectangle around the license plate on the original image
    x, y, w, h = coords
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
    # Display the original image and the detected plate
    cv2.imshow("Original Image with Plate Detected", image)
    cv2.imshow("Detected License Plate", plate)
    
    # Save the detected plate
    output_path = "detected_plate.jpg"
    cv2.imwrite(output_path, plate)
    print(f"Detected license plate saved as '{output_path}'")
    
    # Wait for just 5 seconds (5000 milliseconds)
    cv2.waitKey(5000)
    cv2.destroyAllWindows()