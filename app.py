# app.py

from flask import Flask, request, render_template
import os
from core.core_pipeline import process_license_plate
from malaysia_plate_info import identify_state

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
RESULT_FOLDER = 'static/results'

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['image']
        if file:
            filename = file.filename
            upload_path = os.path.join(UPLOAD_FOLDER, filename)
            file.save(upload_path)

            result = process_license_plate(upload_path)

            if 'error' in result:
                ocr_output = ''
                cleaned_output = ''
            else:
                ocr_output = result['raw_ocr']
                cleaned_output = result['cleaned_output']

            state, plate_type = identify_state(cleaned_output)
            processing_images = result.get('processing_images', [])
            candidates = result.get('candidates', [])

            return render_template('index.html',
                                   image_file=filename,
                                   ocr_output=ocr_output,
                                   cleaned_output=cleaned_output,
                                   state=state,
                                   plate_type=plate_type,
                                   processing_images=processing_images,
                                   candidates=candidates)

    return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True)