import cv2
import numpy as np
import pytesseract
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk
import os
import imutils
from typing import Optional, Tuple, List, Any

# Suppress Tk deprecation warning
os.environ['TK_SILENCE_DEPRECATION'] = '1'

# Malaysian state and special plate prefixes (simplified)
STATE_PREFIXES = {
    'A': 'Perak', 'B': 'Selangor', 'C': 'Pahang', 'D': 'Kelantan', 'F': 'Putrajaya',
    'J': 'Johor', 'K': 'Kedah', 'M': 'Melaka', 'N': 'Negeri Sembilan', 'P': 'Penang',
    'Q': 'Sarawak', 'R': 'Perlis', 'T': 'Terengganu', 'V': 'Kuala Lumpur', 'W': 'Kuala Lumpur',
    'Z': 'Military', 'H': 'Taxi', 'L': 'Labuan', 'U': 'Sabah',
}

SPECIAL_PREFIXES = {
    'Z': 'Military', 'H': 'Taxi', 'Q': 'Sarawak', 'U': 'Sabah', 'L': 'Labuan',
}

class LicensePlateDetector:
    def __init__(self) -> None:
        self.window = tk.Tk()
        self.window.title("Malaysian License Plate Detector")
        self.window.geometry("1000x700")
        self.setup_gui()
        
    def setup_gui(self) -> None:
        main_frame = ttk.Frame(self.window, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Tabs for each processing step
        self.notebook = ttk.Notebook(main_frame)
        self.notebook.grid(row=0, column=0, columnspan=3, pady=10, sticky='nsew')
        
        self.img_labels = []
        self.img_titles = [
            "Original", "Grayscale", "Bilateral Filter", "Canny Edges", "Contours", "Plate Region"
        ]
        for title in self.img_titles:
            frame = ttk.Frame(self.notebook)
            label = ttk.Label(frame)
            label.pack()
            self.img_labels.append(label)
            self.notebook.add(frame, text=title)

        # Buttons
        ttk.Button(main_frame, text="Select Image", command=self.load_image).grid(row=1, column=0, pady=5)
        ttk.Button(main_frame, text="Detect License Plate", command=self.process_image).grid(row=1, column=1, pady=5)
        
        # Results
        self.result_label = ttk.Label(main_frame, text="Results will appear here", font=("Arial", 14))
        self.result_label.grid(row=2, column=0, columnspan=3, pady=10)
        
        self.current_image: Optional[np.ndarray] = None
        self.image_steps: List[Optional[np.ndarray]] = [None] * len(self.img_titles)
        self.plate_img: Optional[np.ndarray] = None

    def load_image(self) -> None:
        file_path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.gif *.tiff")]
        )
        if file_path:
            self.current_image = cv2.imread(file_path)
            if self.current_image is not None:
                self.current_image = imutils.resize(self.current_image, width=500)
                self.image_steps[0] = self.current_image.copy()
                self.update_step_images()
                self.result_label.config(text="Image loaded. Click 'Detect License Plate' to proceed.")

    def update_step_images(self):
        for idx, img in enumerate(self.image_steps):
            if img is not None:
                img_disp = img.copy()
                if len(img_disp.shape) == 2:
                    img_disp = cv2.cvtColor(img_disp, cv2.COLOR_GRAY2RGB)
                img_disp = imutils.resize(img_disp, width=400)
                img_pil = Image.fromarray(img_disp)
                img_tk = ImageTk.PhotoImage(img_pil)
                self.img_labels[idx].configure(image=img_tk)
                self.img_labels[idx].image = img_tk
            else:
                self.img_labels[idx].configure(image=None)
                self.img_labels[idx].image = None

    def process_image(self) -> None:
        if self.current_image is None:
            messagebox.showerror("Error", "Please load an image first!")
            return
        img = self.current_image.copy()
        self.image_steps[0] = img.copy()
        # Step 1: Grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        self.image_steps[1] = gray.copy()
        # Step 2: Bilateral Filter
        filtered = cv2.bilateralFilter(gray, 11, 17, 17)
        self.image_steps[2] = filtered.copy()
        # Step 3: Canny Edges
        edged = cv2.Canny(filtered, 170, 200)
        self.image_steps[3] = edged.copy()
        # Step 4: Contours
        contours, _ = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:30]
        plate_contour = None
        for contour in contours:
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
            if len(approx) == 4:
                plate_contour = approx
                break
        contour_img = img.copy()
        if plate_contour is not None:
            cv2.drawContours(contour_img, [plate_contour], -1, (0,255,0), 3)
        self.image_steps[4] = contour_img.copy()
        # Step 5: Plate Region
        plate_img = None
        plate_text = ""
        state_info = ""
        if plate_contour is not None:
            rect = cv2.minAreaRect(plate_contour)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            x, y, w, h = cv2.boundingRect(box)
            plate_img = img[y:y+h, x:x+w]
            if plate_img.size > 0:
                self.image_steps[5] = plate_img.copy()
                # Preprocess for OCR
                ocr_img = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
                ocr_img = cv2.threshold(ocr_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
                # OCR
                plate_text = pytesseract.image_to_string(ocr_img, config='--psm 7').strip().replace("\n", " ")
                state_info = self.classify_plate(plate_text)
            else:
                self.image_steps[5] = np.zeros((50,200,3), dtype=np.uint8)
        else:
            self.image_steps[5] = np.zeros((50,200,3), dtype=np.uint8)
        self.update_step_images()
        # Show result
        if plate_text:
            self.result_label.config(text=f"Detected Plate: {plate_text}\n{state_info}")
        else:
            self.result_label.config(text="No license plate detected or recognized.")

    def classify_plate(self, text: str) -> str:
        # Clean up text
        text = text.replace(" ", "").upper()
        if not text:
            return "Could not classify plate."
        # Check for special plates
        for prefix, desc in SPECIAL_PREFIXES.items():
            if text.startswith(prefix):
                return f"Special Plate: {desc}"
        # Check for state
        if text[0] in STATE_PREFIXES:
            return f"State: {STATE_PREFIXES[text[0]]}"
        return "Unknown or unclassified plate."

    def run(self) -> None:
        self.window.mainloop()

if __name__ == "__main__":
    detector = LicensePlateDetector()
    detector.run() 