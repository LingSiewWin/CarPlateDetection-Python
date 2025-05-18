# Malaysia License Plate Recognition System

A web-based system for recognizing Malaysian vehicle license plates using Flask, OpenCV, and EasyOCR. The system features robust OCR postprocessing to correct common recognition errors and a modern, user-friendly interface.

## Features

- **Automatic detection and recognition** of Malaysian license plates from images
- **Robust OCR postprocessing**:
  - Corrects common OCR mistakes in both the prefix (alphabet) and number sections
  - Combines split OCR results (e.g., prefix and number detected separately)
  - Always prefers a single valid plate candidate if available
- **Modern web UI**:
  - Large, easy-to-use file upload and preview
  - Step-by-step preprocessing visualization
  - Candidate plate information table
  - Final recognition result with state identification
- **Supports correction mappings**:
  - Prefix: `0→O`, `1→I`, `2→Z`, `5→S`, `6→G`, `8→B`
  - Number: `B→8`, `S→5`, `I→1`, `O→0`, `Z→2`, `G→6`

## Setup

1. **Clone the repository:**
   ```bash
   git clone <repo-url>
   cd CarPlateDetection-Python
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
   (Make sure you have OpenCV, Flask, EasyOCR, and numpy installed.)

3. **Run the app:**
   ```bash
   python app.py
   ```
   The app will be available at `http://127.0.0.1:5000/`.

## Usage

1. Open the web interface in your browser.
2. Upload an image containing a Malaysian license plate.
3. Preview the uploaded image.
4. Click the **Detect** button to start recognition.
5. View preprocessing steps, candidate results, and the final recognized plate and state.

## Example

- **Input:**
  - Image of a car with plate `CDK7489`
- **Output:**
  - Plate Number: `CDK7489`
  - Registered State: (e.g., Johor)

## OCR Postprocessing Logic

- If any candidate matches the full plate format, it is used as the result.
- If not, the system tries to combine split candidates (prefix + number), applying correction mappings.
- The number section is always cleaned so that any letter (e.g., `Z`, `B`, `S`) is converted to its most likely digit (`2`, `8`, `5`, etc.).
- The prefix is always cleaned so that any digit (e.g., `0`, `1`) is converted to its most likely letter (`O`, `I`, etc.).

## Notes

- The system is designed for Malaysian plate formats but can be extended for other regions.
- For best results, use clear, high-resolution images of license plates.

## License

MIT License 