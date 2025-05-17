import cv2
import numpy as np
import easyocr
import os

#function to get the state name from the state code
def get_malaysian_state(state_code):
    states = {
        'A': 'Perak',
        'B': 'Selangor',
        'C': 'Pahang',
        'D': 'Kelantan',
        'F': 'Putrajaya',
        'J': 'Johor',
        'K': 'Kedah',
        'M': 'Melaka',
        'N': 'Negeri Sembilan',
        'P': 'Penang',
        'R': 'Perlis',
        'T': 'Terengganu',
        'V': 'Kuala Lumpur',
        'W': 'Kuala Lumpur',
        'S': 'Sabah',
        'Q': 'Sarawak',
    }
    return states.get(state_code, 'Unknown State')

#function to detect the license plate and extract the text
def detect_license_plate(image_path):
    # Initialize EasyOCR reader
    reader = easyocr.Reader(['en'])
    
    # Read the image
    image = cv2.imread(image_path)
    # Create a larger canvas to display information on the side
    height, width = image.shape[:2]
    canvas = np.zeros((height, width + 300, 3), dtype=np.uint8)
    canvas[:, :width] = image
    canvas[:, width:] = (255, 255, 255)  # White background for text area
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply noise reduction and edge detection
    bfilter = cv2.bilateralFilter(gray, 11, 17, 17)
    edged = cv2.Canny(bfilter, 30, 200)
    
    # Find contours
    contours, _ = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
    
    location = None
    for contour in contours:
        approx = cv2.approxPolyDP(contour, 10, True)
        if len(approx) == 4:
            location = approx
            break
    
    if location is None:
        print("No license plate detected")
        return None, None
    
    # Create a mask and extract the license plate
    mask = np.zeros(gray.shape, np.uint8)
    plate_image = cv2.drawContours(mask, [location], 0, 255, -1)
    plate_image = cv2.bitwise_and(image, image, mask=mask)
    
    # Get the coordinates of the license plate
    (x, y) = np.where(mask == 255)
    (x1, y1) = (np.min(y), np.min(x))
    (x2, y2) = (np.max(y), np.max(x))
    cropped_plate = gray[y1:y2+1, x1:x2+1]
    
    # Read the license plate text
    result = reader.readtext(cropped_plate)
    
    # Draw rectangle around plate and display text
    if result:
        # Combine all detected text lines into one row
        text = ''.join([detection[1].replace(' ', '') for detection in result])
        
        # Draw red rectangle around the license plate
        cv2.rectangle(canvas[:, :width], (x1, y1), (x2, y2), (0, 0, 255), 2)
        
        # Get the first letter for state identification
        first_letter = text[0] if text else ''
        state_name = get_malaysian_state(first_letter)
        
        # Calculate coordinates for the circle around first letter
        letter_width = (x2 - x1) // len(text)
        circle_center = (x1 + letter_width // 2, (y1 + y2) // 2)
        circle_radius = letter_width // 2
        
        # Draw yellow circle around first letter
        cv2.circle(canvas[:, :width], circle_center, circle_radius, (0, 255, 255), 2)
        
        # Display information on the side panel
        info_x = width + 10
        cv2.putText(canvas, "License Plate Info:", (info_x, 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        cv2.putText(canvas, f"Number: {text}", (info_x, 80), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        cv2.putText(canvas, f"State: {state_name}", (info_x, 120), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        
        return canvas, text, state_name
    
    return canvas, None, None

def main():
    # Path to the test folder
    test_folder = 'successful_motorbike' #change folder name to choose which folder to run through
    
    # Get all jpg files in the test folder
    image_files = [f for f in os.listdir(test_folder) if f.endswith('.jpg')]
    
    # Process each image
    for image_file in image_files:
        image_path = os.path.join(test_folder, image_file)
        print(f"\nProcessing: {image_file}")
        
        processed_image, plate_text, state = detect_license_plate(image_path)
        
        if processed_image is not None:
            if plate_text:
                print(f"Detected license plate: {plate_text}")
                if state:
                    print(f"Vehicle registered in: {state}")
            else:
                print("License plate detected but text could not be read")
                
            # Display the result
            cv2.imshow('License Plate Detection', processed_image)
            cv2.waitKey(1000)  # Display each image for 1 second
    
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()