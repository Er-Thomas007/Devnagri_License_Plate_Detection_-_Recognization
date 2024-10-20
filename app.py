from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
import cv2
import numpy as np
import base64
from io import BytesIO
from ultralytics import YOLO

app = Flask(__name__)

# Load YOLO models
license_plate_model = YOLO(r'D:\AI_Project\Advanced_Recognization_of_Nepali_License_Plates_using_YOLOv8_and_OCR_Technologies\DeepLearning_Models\best24.pt')
character_model = YOLO(r'D:\AI_Project\Advanced_Recognization_of_Nepali_License_Plates_using_YOLOv8_and_OCR_Technologies\DeepLearning_Models\best.pt')

# Nepali labels for character recognition
nepali_labels = {
    "0": "०", "1": "१", "2": "२", "3": "३", "4": "४", "5": "५", "6": "६", "7": "७", "8": "८", "9": "९",
    "Bagmati": "बागमती", "CHA": "च", "JA": "ज", "KA": "क", "KHA": "ख", "Pradesh": "प्रदेश", "JHA": "झ",
    "p": "प", "PRA": "प्र", "SA": "स", "YA": "य", "BA": "बा"
}

def process_image(img):
    """Process the input image using the YOLO model for license plates."""
    results = license_plate_model.predict(img)
    return results

def detect_characters(license_plate_region):
    """Detect characters within the license plate region."""
    char_results = character_model.predict(license_plate_region)
    detected_characters = []
    for char_result in char_results:
        for char_box in char_result.boxes:
            char_cls = int(char_box.cls.item())
            char_conf = float(char_box.conf.item())
            char_label = character_model.names[char_cls]
            char_x1, char_y1, char_x2, char_y2 = map(int, char_box.xyxy[0].tolist())
            detected_characters.append((char_label, char_x1, char_y1))
    return detected_characters

def display_detected_characters(detected_characters):
    """Format detected characters for output."""
    if not detected_characters:
        return "", ""

    detected_characters.sort(key=lambda x: (x[2], x[1]))

    rows = []
    current_row = []
    previous_y = detected_characters[0][2]

    for label, x, y in detected_characters:
        if abs(y - previous_y) > 20:  # Adjust this threshold as necessary
            rows.append(current_row)
            current_row = []
        current_row.append((label, x))
        previous_y = y

    if current_row:
        rows.append(current_row)

    english_output = ""
    nepali_output = ""

    for row in rows:
        row.sort(key=lambda x: x[1])  # Sort by x-coordinate
        reversed_row = list((row))
        english_row = " ".join(label for label, _ in reversed_row)
        nepali_row = " ".join(nepali_labels.get(label, label) for label, _ in reversed_row)
        english_output += f"{english_row}\n"
        nepali_output += f"{nepali_row}\n"

    return english_output.strip(), nepali_output.strip()

def draw_boxes_and_characters(img, results):
    """Draw bounding boxes and detected characters on the image."""
    detected_characters = []
    is_bagmati = False
    
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            cls = int(box.cls.item())
            label = license_plate_model.names[cls]
            conf = float(box.conf.item())
            
            if cls == 1:  # License plate class
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green box
                cv2.putText(img, f"{label} {conf:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                license_plate_region = img[y1:y2, x1:x2]
                zoomed_license_plate = cv2.resize(license_plate_region, (224, 224))
                char_results = character_model.predict(zoomed_license_plate)

                for char_result in char_results:
                    for char_box in char_result.boxes:
                        char_cls = int(char_box.cls.item())
                        char_conf = float(char_box.conf.item())
                        char_label = character_model.names[char_cls]
                        char_x1, char_y1, char_x2, char_y2 = map(int, char_box.xyxy[0].tolist())

                        orig_char_x1 = x1 + char_x1 * ((x2 - x1) / 224)
                        orig_char_y1 = y1 + char_y1 * ((y2 - y1) / 224)
                        orig_char_x2 = x1 + char_x2 * ((x2 - x1) / 224)
                        orig_char_y2 = y1 + char_y2 * ((y2 - y1) / 224)

                        detected_characters.append((char_label, orig_char_x1, orig_char_y1))

                        cv2.rectangle(img, (int(orig_char_x1), int(orig_char_y1)), (int(orig_char_x2), int(orig_char_y2)), (0, 255, 0), 2)
                        cv2.putText(img, char_label, (int(orig_char_x1), int(orig_char_y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                # Check for "Bagmati" or "BA" in the detected characters
                detected_labels = [char_label for char_label, _, _ in detected_characters]
                if "Bagmati" in detected_labels or "BA" in detected_labels:
                    is_bagmati = True

    return img, detected_characters, is_bagmati

def image_to_base64(img):
    """Convert image to base64 encoding."""
    _, img_encoded = cv2.imencode('.jpg', img)
    img_bytes = img_encoded.tobytes()
    return base64.b64encode(img_bytes).decode('utf-8')

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        if 'file' not in request.files:
            return 'No file part', 400
        file = request.files['file']
        if file.filename == '':
            return 'No selected file', 400
        if file:
            img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
            results = process_image(img)
            img_with_boxes, detected_characters, is_bagmati = draw_boxes_and_characters(img, results)
            english_text, nepali_text = display_detected_characters(detected_characters)
            img_base64 = image_to_base64(img_with_boxes)
            return render_template('result.html', 
                                   input_img_data=image_to_base64(img), 
                                   processed_img_data=img_base64, 
                                   english_text=english_text, 
                                   nepali_text=nepali_text,
                                   is_bagmati=is_bagmati)
    return render_template('upload.html')


@app.route('/capture')
def capture():
    return render_template('capture.html')

@app.route('/process_image_route', methods=['POST'])
def process_image_route():
    data = request.get_json()
    image_data = data.get('image')

    if image_data.startswith('data:image/png;base64,'):
        image_data = image_data.replace('data:image/png;base64,', '')

    image_bytes = base64.b64decode(image_data)
    img = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
    
    # Process the image with YOLO model
    results = process_image(img)
    img_with_boxes, detected_characters, is_bagmati = draw_boxes_and_characters(img, results)
    
    # Get detected characters and convert to English and Nepali text
    english_text, nepali_text = display_detected_characters(detected_characters)
    
    # Convert processed image to base64
    img_base64 = image_to_base64(img_with_boxes)
    
    return jsonify({
        'processed_image': img_base64,
        'english_text': english_text,
        'nepali_text': nepali_text,
        'is_bagmati': is_bagmati
    })



@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/faqs')
def faqs():
    return render_template('faqs.html')

@app.route('/support')
def support():
    return render_template('support.html')

@app.route('/privacy-policy')
def privacy_policy():
    return render_template('privacy_policy.html')

@app.route('/privacy-policy-full')
def privacy_policy_full():
    return render_template('privacy_policy_full.html')

if __name__ == '__main__':
    app.run(debug=True)
