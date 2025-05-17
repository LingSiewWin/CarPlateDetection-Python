from flask import Flask, request, jsonify, send_file, render_template, url_for, request as flask_request
import os
import cv2
import numpy as np
import pytesseract
import imutils
from PIL import Image, ImageEnhance
from uuid import uuid4
import re
from malaysia_plate_info import identify_state
import difflib

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
RESULT_FOLDER = 'results'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

# Malaysian plate regex: 1-3 letters + 1-4 numbers + optional 1 letter
PLATE_REGEX = re.compile(r'([A-Z]{1,3})\s*([0-9]{1,4})\s*([A-Z]?)')

# Loosened filtering parameters
ASPECT_RATIO_RANGE = (1.5, 10.0)
AREA_RANGE = (1000, 100000)
EDGE_DENSITY_RANGE = (0.005, 0.5)
INTENSITY_STD_MIN = 10
TOP_N_CANDIDATES = 5

# Relative plate width/vehicle width ratio ranges for each vehicle type
RELATIVE_WIDTH_RANGES = {
    'van': (0.10, 0.30),      # Plate is 10% to 30% of vehicle width
    'pickup': (0.08, 0.25),  # Plate is 8% to 25% of vehicle width
    'default': (0.07, 0.35)  # Fallback for unknown type
}

# Expected plate size in upscaled ROI (tune as needed)
EXPECTED_W, EXPECTED_H = 180, 40

def order_points(pts):
    # Order points: top-left, top-right, bottom-right, bottom-left
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def four_point_transform(image, pts):
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    maxWidth = max(int(widthA), int(widthB))
    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxHeight = max(int(heightA), int(heightB))
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    return warped

# Add new function for OCR preprocessing
def preprocess_ocr_text(text):
    # Remove extra spaces, common OCR mistakes
    text = text.upper().replace(' ', '')
    # Common OCR misreads
    text = text.replace('0', 'O')  # or reverse if needed
    text = text.replace('1', 'I')
    text = text.replace('8', 'B')
    text = text.replace('5', 'S')
    text = text.replace('6', 'G')
    text = text.replace('2', 'Z')
    # Add more replacements as needed
    return text

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    # Get vehicle type from query parameter or form (default to 'default')
    vehicle_type = flask_request.args.get('vehicle_type', 'default').lower()
    rel_width_min, rel_width_max = RELATIVE_WIDTH_RANGES.get(vehicle_type, RELATIVE_WIDTH_RANGES['default'])
    # Get ROI mode (lower or full)
    roi_mode = flask_request.args.get('roi_mode', 'lower').lower()  # 'lower' or 'full'
    if 'image' not in request.files:
        return jsonify({'status': 'fail', 'message': 'No file uploaded.'})
    file = request.files['image']
    if file.filename == '':
        return jsonify({'status': 'fail', 'message': 'No file selected.'})
    uid = str(uuid4())
    img_path = os.path.join(UPLOAD_FOLDER, f'{uid}_{file.filename}')
    file.save(img_path)
    img = cv2.imread(img_path)
    if img is None:
        return jsonify({'status': 'fail', 'message': 'Invalid image.'})
    img = imutils.resize(img, width=500)
    h, w = img.shape[:2]
    # ROI selection
    if roi_mode == 'full':
        roi_img = img
    else:
        roi_img = img[int(h/2):, :]
    roi_img = cv2.resize(roi_img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    roi_h, roi_w = roi_img.shape[:2]
    steps = []
    titles = ["Original", "ROI (Lower 1/2, Upscaled)", "Grayscale", "Bilateral Filter", "Canny Edges", "Contours", "Plate Region", "OCR Input"]
    # Step 0: Original
    orig_path = os.path.join(RESULT_FOLDER, f'step0_{uid}.png')
    cv2.imwrite(orig_path, img)
    steps.append(url_for('static', filename=f'results/step0_{uid}.png'))
    # Step 1: ROI (upscaled)
    roi_path = os.path.join(RESULT_FOLDER, f'step1_{uid}.png')
    cv2.imwrite(roi_path, roi_img)
    steps.append(url_for('static', filename=f'results/step1_{uid}.png'))
    # Step 2: Grayscale
    gray = cv2.cvtColor(roi_img, cv2.COLOR_BGR2GRAY)
    gray_path = os.path.join(RESULT_FOLDER, f'step2_{uid}.png')
    cv2.imwrite(gray_path, gray)
    steps.append(url_for('static', filename=f'results/step2_{uid}.png'))
    # Step 3: Bilateral Filter
    filtered = cv2.bilateralFilter(gray, 11, 17, 17)
    filt_path = os.path.join(RESULT_FOLDER, f'step3_{uid}.png')
    cv2.imwrite(filt_path, filtered)
    steps.append(url_for('static', filename=f'results/step3_{uid}.png'))
    # Step 4: Canny Edges
    edged = cv2.Canny(filtered, 170, 200)
    edge_path = os.path.join(RESULT_FOLDER, f'step4_{uid}.png')
    cv2.imwrite(edge_path, edged)
    steps.append(url_for('static', filename=f'results/step4_{uid}.png'))
    # Step 5: Contours (use hierarchical)
    contours, hierarchy = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    print(f"Total contours found: {len(contours)}")
    # Relaxed geometric filters
    ASPECT_RATIO_RANGE = (1.0, 12.0)  # Allow more skewed plates
    ANGLE_MAX = 85  # Allow more rotation/skew
    candidates = []
    for contour in contours:
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
        if len(approx) < 4 or len(approx) > 6:
            continue
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        x, y, w_box, h_box = cv2.boundingRect(box)
        aspect_ratio = w_box / float(h_box) if h_box > 0 else 0
        area = w_box * h_box
        angle = abs(rect[2])
        if not (ASPECT_RATIO_RANGE[0] <= aspect_ratio <= ASPECT_RATIO_RANGE[1]):
            continue
        if not (AREA_RANGE[0] <= area <= AREA_RANGE[1]):
            continue
        if angle > ANGLE_MAX:
            continue
        rel_width = w_box / float(roi_w)
        if not (rel_width_min <= rel_width <= rel_width_max):
            continue
        roi = edged[y:y+h_box, x:x+w_box]
        edge_density = np.sum(roi > 0) / float(w_box * h_box)
        if not (EDGE_DENSITY_RANGE[0] <= edge_density <= EDGE_DENSITY_RANGE[1]):
            continue
        roi_gray = gray[y:y+h_box, x:x+w_box]
        mean_intensity = np.mean(roi_gray)
        std_intensity = np.std(roi_gray)
        if std_intensity < INTENSITY_STD_MIN:
            continue
        candidates.append((contour, box, edge_density, std_intensity, area, (x, y, w_box, h_box)))
    print(f"Candidates after all filters: {len(candidates)}")
    candidates = sorted(candidates, key=lambda x: -x[4])[:TOP_N_CANDIDATES]
    best_plate = ""
    best_classification = ""
    best_match_score = 0
    best_plate_img = None
    best_ocr_img = None
    best_steps = None
    best_raw_ocr = ""
    best_pre_ocr = ""
    contour_img = roi_img.copy()
    for idx, (contour, box, edge_density, std_intensity, area, bbox) in enumerate(candidates):
        warped_plate = four_point_transform(roi_img, box)  # Always apply perspective transform
        plate_img_up = cv2.resize(warped_plate, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        pil_img = Image.fromarray(cv2.cvtColor(plate_img_up, cv2.COLOR_BGR2RGB))
        pil_img = ImageEnhance.Contrast(pil_img).enhance(2.0)
        pil_img = ImageEnhance.Sharpness(pil_img).enhance(2.0)
        ocr_img = np.array(pil_img)
        ocr_img = cv2.cvtColor(ocr_img, cv2.COLOR_RGB2GRAY)
        ocr_img_adapt = cv2.adaptiveThreshold(ocr_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        raw_text = pytesseract.image_to_string(ocr_img_adapt, config='--psm 7').strip().replace("\n", " ")
        preprocessed_text = preprocess_ocr_text(raw_text)
        match = PLATE_REGEX.search(preprocessed_text)
        match_score = 0
        if match:
            match_score = 1
        else:
            ocr_img_otsu = cv2.threshold(ocr_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
            raw_text2 = pytesseract.image_to_string(ocr_img_otsu, config='--psm 7').strip().replace("\n", " ")
            preprocessed_text2 = preprocess_ocr_text(raw_text2)
            match = PLATE_REGEX.search(preprocessed_text2)
            if match:
                match_score = 1
                raw_text = raw_text2
                preprocessed_text = preprocessed_text2
        if match_score > best_match_score:
            best_match_score = match_score
            best_plate = ' '.join([g for g in match.groups() if g]) if match else preprocessed_text
            state, plate_type = identify_state(best_plate)
            best_classification = f"{state} ({plate_type})"
            best_plate_img = plate_img_up
            best_ocr_img = ocr_img_adapt
            best_raw_ocr = raw_text
            best_pre_ocr = preprocessed_text
            contour_img = roi_img.copy()
            cv2.drawContours(contour_img, [box], -1, (0,255,0), 3)
    # If no good candidate, fallback to largest contour
    if not candidates and len(contours) > 0:
        print("No candidate passed all filters, using expected size closeness fallback.")
        fallback_boxes = []
        for cnt in contours:
            x, y, w_box, h_box = cv2.boundingRect(cnt)
            aspect_ratio = w_box / float(h_box) if h_box > 0 else 0
            area = w_box * h_box
            if 2 < aspect_ratio < 6 and 1000 < area < 100000:
                fallback_boxes.append((x, y, w_box, h_box))
        if fallback_boxes:
            best_box = sorted(fallback_boxes, key=lambda b: abs(b[2] - EXPECTED_W) + abs(b[3] - EXPECTED_H))[0]
            x, y, w_box, h_box = best_box
            plate_img = roi_img[y:y+h_box, x:x+w_box]
            if plate_img.size > 0:
                plate_img_up = cv2.resize(plate_img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
                pil_img = Image.fromarray(cv2.cvtColor(plate_img_up, cv2.COLOR_BGR2RGB))
                pil_img = ImageEnhance.Contrast(pil_img).enhance(2.0)
                pil_img = ImageEnhance.Sharpness(pil_img).enhance(2.0)
                ocr_img = np.array(pil_img)
                ocr_img = cv2.cvtColor(ocr_img, cv2.COLOR_RGB2GRAY)
                ocr_img_adapt = cv2.adaptiveThreshold(ocr_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
                raw_text = pytesseract.image_to_string(ocr_img_adapt, config='--psm 7').strip().replace("\n", " ")
                preprocessed_text = preprocess_ocr_text(raw_text)
                match = PLATE_REGEX.search(preprocessed_text)
                if match:
                    best_plate = ' '.join([g for g in match.groups() if g])
                    state, plate_type = identify_state(best_plate)
                    best_classification = f"{state} ({plate_type})"
                else:
                    best_plate = preprocessed_text
                    best_classification = "Could not classify plate."
                best_plate_img = plate_img_up
                best_ocr_img = ocr_img_adapt
                best_raw_ocr = raw_text
                best_pre_ocr = preprocessed_text
                contour_img = roi_img.copy()
                cv2.rectangle(contour_img, (x, y), (x + w_box, y + h_box), (0, 255, 0), 2)
    cont_path = os.path.join(RESULT_FOLDER, f'step5_{uid}.png')
    cv2.imwrite(cont_path, contour_img)
    steps.append(url_for('static', filename=f'results/step5_{uid}.png'))
    plate_path = os.path.join(RESULT_FOLDER, f'step6_{uid}.png')
    ocr_input_path = os.path.join(RESULT_FOLDER, f'step7_{uid}.png')
    if best_plate_img is not None:
        cv2.imwrite(plate_path, best_plate_img)
        cv2.imwrite(ocr_input_path, best_ocr_img)
        steps.append(url_for('static', filename=f'results/step6_{uid}.png'))
        steps.append(url_for('static', filename=f'results/step7_{uid}.png'))
    else:
        cv2.imwrite(plate_path, np.zeros((50,200,3), dtype=np.uint8))
        cv2.imwrite(ocr_input_path, np.zeros((50,200), dtype=np.uint8))
        steps.append(url_for('static', filename=f'results/step6_{uid}.png'))
        steps.append(url_for('static', filename=f'results/step7_{uid}.png'))
    return jsonify({
        'status': 'success',
        'steps': steps,
        'titles': titles,
        'plate': best_plate,
        'classification': best_classification if best_plate else "Could not classify plate.",
        'raw_ocr': best_raw_ocr,
        'preprocessed_ocr': best_pre_ocr,
        'roi_mode': roi_mode
    })

app.static_folder = '.'

if __name__ == '__main__':
    app.run(debug=True) 