# core_pipeline.py

import cv2
import os
import numpy as np
from core.color_segmentation import find_malaysia_plates
from core.preprocessing import enhance_characters
from core.char_segmentation import segment_characters
from core.ocr_engine import recognize_character, postprocess_ocr_result

def process_license_plate(image_path):
    """
    Complete license plate recognition pipeline, saving intermediate results at each step:
    1. Original image
    2. Grayscale
    3. CLAHE Enhanced
    4. Bilateral Filtered
    5. Anisotropic Diffusion
    6. Morphological Gradient
    7. Canny Edges
    8. Morphed (Noise Reduced)
    9. Candidate Contours
    10. Final Annotated Image
    """
    result_dir = "static/results"
    os.makedirs(result_dir, exist_ok=True)

    # 1. Load image
    img = cv2.imread(image_path)
    if img is None:
        return {"error": "Unable to load image"}

    step_images = []
    def save_step(label, img_data):
        fname = f"step_{len(step_images)}_{label.replace(' ', '_').lower()}.jpg"
        fpath = os.path.join(result_dir, fname)
        cv2.imwrite(fpath, img_data)
        step_images.append((label, fname))
        return fname

    save_step("Original Image", img)

    # 2. Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    save_step("Grayscale", gray)

    # 3. CLAHE Enhanced
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe_img = clahe.apply(gray)
    save_step("CLAHE Enhanced", clahe_img)

    # 4. Bilateral Filtered
    bilateral = cv2.bilateralFilter(clahe_img, 11, 17, 17)
    save_step("Bilateral Filtered", bilateral)

    # 5. Anisotropic Diffusion (simple approx: edge-preserving filter)
    try:
        diffused = cv2.edgePreservingFilter(bilateral, flags=1, sigma_s=60, sigma_r=0.4)
        if len(diffused.shape) == 3:
            diffused = cv2.cvtColor(diffused, cv2.COLOR_BGR2GRAY)
    except Exception:
        diffused = bilateral.copy()
    save_step("Anisotropic Diffusion", diffused)

    # 6. Morphological Gradient
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    morph_grad = cv2.morphologyEx(diffused, cv2.MORPH_GRADIENT, kernel)
    save_step("Morphological Gradient", morph_grad)

    # 7. Canny Edges
    canny = cv2.Canny(morph_grad, 100, 200)
    save_step("Canny Edges", canny)

    # 8. Morphed (Noise Reduced)
    morphed = cv2.morphologyEx(canny, cv2.MORPH_CLOSE, kernel, iterations=2)
    save_step("Morphed (Noise Reduced)", morphed)

    # 9. Candidate Contours (original pipeline)
    contours, _ = cv2.findContours(morphed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour_img = img.copy()
    candidates = []
    for i, cnt in enumerate(contours):
        x, y, w, h = cv2.boundingRect(cnt)
        aspect = w / float(h)
        area = w * h
        if 2 < aspect < 7 and 1000 < area < 50000:
            cv2.rectangle(contour_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            candidates.append((x, y, w, h))
    save_step("Candidate Contours", contour_img)

    # 10. Final Annotated Image (pick best candidate)
    annotated = img.copy()
    candidate_info = []
    best_ocr = ""
    best_bbox = None
    best_score = 0
    for idx, (x, y, w, h) in enumerate(candidates):
        plate_crop = gray[y:y+h, x:x+w]
        # Resize for better OCR
        plate_crop = cv2.resize(plate_crop, (w*3, h*3), interpolation=cv2.INTER_CUBIC)
        # Threshold for OCR
        _, plate_bin = cv2.threshold(plate_crop, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # OCR
        ocr_result = postprocess_ocr_result(recognize_character(plate_bin))
        candidate_info.append({
            "index": idx+1,
            "bbox": (x, y, w, h),
            "ocr": ocr_result
        })
        # Save candidate crop
        crop_fname = f"candidate_{idx+1}_at_{x}_{y}.jpg"
        cv2.imwrite(os.path.join(result_dir, crop_fname), plate_bin)
        # Annotate
        cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.putText(annotated, f"{ocr_result}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
        # Pick best by length and alnum
        score = len([c for c in ocr_result if c.isalnum()])
        if score > best_score:
            best_score = score
            best_ocr = ocr_result
            best_bbox = (x, y, w, h)
    if best_bbox:
        x, y, w, h = best_bbox
        cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 255, 255), 3)
        cv2.putText(annotated, f"Best: {best_ocr}", (x, y+h+30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)
    save_step("Final Annotated Image", annotated)

    # --- Parallel: Relaxed 4-corner contour detection ---
    parallel_candidates = []
    parallel_best_ocr = ""
    parallel_best_score = 0
    parallel_best_bbox = None
    parallel_results = []
    # Use RETR_TREE to get all contours
    all_contours, _ = cv2.findContours(morphed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    all_contours = sorted(all_contours, key=cv2.contourArea, reverse=True)[:10]
    for idx, contour in enumerate(all_contours):
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
        if len(approx) == 4:
            mask = np.zeros(gray.shape, np.uint8)
            cv2.drawContours(mask, [approx], 0, 255, -1)
            plate_img = cv2.bitwise_and(img, img, mask=mask)
            (x, y) = np.where(mask == 255)
            if len(x) == 0 or len(y) == 0:
                continue
            (x1, y1) = (np.min(y), np.min(x))
            (x2, y2) = (np.max(y), np.max(x))
            cropped_plate = gray[y1:y2+1, x1:x2+1]
            if cropped_plate.size == 0:
                continue
            # Resize for better OCR
            plate_crop = cv2.resize(cropped_plate, (max(60, (x2-x1)*3), max(20, (y2-y1)*3)), interpolation=cv2.INTER_CUBIC)
            # Threshold for OCR
            _, plate_bin = cv2.threshold(plate_crop, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            ocr_result = postprocess_ocr_result(recognize_character(plate_bin))
            # Validate with strict format
            from core.char_segmentation import is_valid_plate_format
            if is_valid_plate_format(ocr_result):
                parallel_results.append({
                    "index": idx+1,
                    "bbox": (x1, y1, x2-x1, y2-y1),
                    "ocr": ocr_result
                })
                score = len([c for c in ocr_result if c.isalnum()])
                if score > parallel_best_score:
                    parallel_best_score = score
                    parallel_best_ocr = ocr_result
                    parallel_best_bbox = (x1, y1, x2-x1, y2-y1)
    # If no valid, still show the best (for debug)
    if not parallel_results and all_contours:
        for idx, contour in enumerate(all_contours):
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
            if len(approx) == 4:
                mask = np.zeros(gray.shape, np.uint8)
                cv2.drawContours(mask, [approx], 0, 255, -1)
                plate_img = cv2.bitwise_and(img, img, mask=mask)
                (x, y) = np.where(mask == 255)
                if len(x) == 0 or len(y) == 0:
                    continue
                (x1, y1) = (np.min(y), np.min(x))
                (x2, y2) = (np.max(y), np.max(x))
                cropped_plate = gray[y1:y2+1, x1:x2+1]
                if cropped_plate.size == 0:
                    continue
                plate_crop = cv2.resize(cropped_plate, (max(60, (x2-x1)*3), max(20, (y2-y1)*3)), interpolation=cv2.INTER_CUBIC)
                _, plate_bin = cv2.threshold(plate_crop, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                ocr_result = postprocess_ocr_result(recognize_character(plate_bin))
                parallel_results.append({
                    "index": idx+1,
                    "bbox": (x1, y1, x2-x1, y2-y1),
                    "ocr": ocr_result
                })
                break
    # --- End parallel branch ---

    # If no candidates, fallback to color segmentation pipeline
    if not candidates:
        plate_candidate = find_malaysia_plates(img)
        save_step("Color Segmentation Fallback", plate_candidate)
        enhanced = enhance_characters(plate_candidate)
        save_step("Enhanced Fallback", enhanced)
        chars = segment_characters(enhanced)
        raw_results = [recognize_character(ch) for ch in chars]
        best_ocr = postprocess_ocr_result(''.join(raw_results))

    # Check for a valid full plate among candidates
    valid_full_plate = None
    for c in candidate_info:
        ocr = c.get('ocr', '').upper()
        if is_valid_plate_format(ocr):
            valid_full_plate = ocr
            break
    if valid_full_plate:
        final_plate = valid_full_plate
    else:
        # Combine candidates if possible
        combined_plate = combine_ocr_candidates(candidate_info)
        final_plate = best_ocr
        if combined_plate:
            final_plate = combined_plate
    # Always clean the final plate before display/state detection
    prefix, number = split_plate(final_plate)
    clean_final_plate = clean_prefix(prefix) + clean_number(number)
    return {
        "raw_ocr": best_ocr,
        "cleaned_output": clean_final_plate,
        "processing_images": step_images,
        "candidates": candidate_info,
        "parallel_candidates": parallel_results,
        "parallel_best_ocr": parallel_best_ocr
    }

def clean_prefix(prefix):
    """Correct common OCR digit mistakes in the prefix (alphabet part)."""
    corrections = {'0': 'O', '1': 'I', '2': 'Z', '5': 'S', '6': 'G', '8': 'B'}
    return ''.join(corrections.get(ch, ch) for ch in prefix)

def clean_number(number):
    """Correct common OCR letter mistakes in the number part."""
    corrections = {'B': '8', 'S': '5', 'I': '1', 'O': '0', 'Z': '2', 'G': '6'}
    return ''.join(corrections.get(ch, ch) for ch in number)

def combine_ocr_candidates(candidate_info):
    """
    Combine OCR results from candidates: if there is one candidate that is all letters and another that is all numbers (or mostly numbers),
    combine them as letters+numbers (e.g., AKE9005). Returns the combined string if found, else None.
    Also applies cleaning to correct common OCR mistakes in both parts.
    """
    import re
    letters = None
    numbers = None
    for c in candidate_info:
        ocr = c.get('ocr', '').upper()
        if len(ocr) < 2:
            continue
        # Accept if all letters
        if re.fullmatch(r'[A-Z]{2,}', ocr):
            letters = ocr
        # Accept if all digits or mostly digits with 1 letter (common OCR mistake)
        elif re.fullmatch(r'[0-9]{2,}[A-Z]?', ocr) or re.fullmatch(r'[0-9A-Z]{2,}', ocr):
            # Accept if at least 2 digits
            digits = sum(1 for ch in ocr if ch.isdigit())
            if digits >= 2:
                numbers = ocr
    if letters and numbers:
        clean_letters = clean_prefix(letters)
        clean_numbers = clean_number(numbers)
        return clean_letters + clean_numbers
    return None

def split_plate(plate):
    """Split a plate string into prefix (letters) and number (digits) parts."""
    import re
    match = re.match(r'^([A-Z]{1,3})([0-9A-Z]{1,4})$', plate)
    if match:
        return match.group(1), match.group(2)
    return plate, ''

def is_valid_plate_format(plate):
    import re
    return re.fullmatch(r'[A-Z]{1,3}[0-9]{1,4}', plate) is not None