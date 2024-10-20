import cv2
import tempfile
import streamlit as st
from ultralytics import YOLO
import numpy as np

nepali_labels = {
    "0": "०", "1": "१", "2": "२", "3": "३", "4": "४", "5": "५", "6": "६", "7": "७", "8": "८", "9": "९",
    "Bagmati": "बागमती", "CHA": "च", "JA": "ज", "KA": "क", "KHA": "ख", "Pradesh": "प्रदेश", "JHA": "झ",
    "p": "प", "PRA": "प्र", "SA": "स", "YA": "य", "BA": "बा"
}

def main():
    st.set_page_config(page_title="License Plate Recognition", page_icon="🚗", layout="wide")

    st.markdown(
        """
        <style>
        .main {
            background-color: #333333; /* Dark background for better contrast */
            padding: 20px;
        }
        .title {
            font-size: 36px;
            font-weight: bold;
            color: #ffffff; /* White color for better visibility */
            text-align: center;
            margin-bottom: 20px;
        }
        .sidebar-title {
            font-size: 24px;
            font-weight: bold;
            color: #000000; /* White color for better visibility */
        }
        .settings-label {
            font-size: 18px;
            font-weight: bold;
            color: #000000; /* White color for better visibility */
            margin-top: 10px;
        }
        .settings-input {
            margin-bottom: 10px;
        }
        .detected-class {
            font-size: 20px;
            color: #e74c3c; /* Red color for detected classes */
            font-weight: bold;
        }
        .nepali-detected-class {
            font-size: 20px;
            color: #2980b9; /* Blue color for Nepali detected classes */
            font-weight: bold;
        }
        .province {
            font-size: 22px;
            color: #27ae60; /* Green color for province */
            font-weight: bold;
            text-align: center;
            margin-top: 20px;
        }
        .number-plate {
            font-size: 24px;
            color: #FFFFFF; /* Golden color */ /* Green color for number plate text */
            font-weight: bold;
            text-align: center;
            margin-top: 10px;
            margin-bottom: 10px;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.markdown('<div class="main">', unsafe_allow_html=True)
    st.markdown('<div class="title">YOLO License Plate Detection and Character Recognition</div>', unsafe_allow_html=True)

    st.sidebar.markdown('<div class="sidebar-title">Settings</div>', unsafe_allow_html=True)

    st.sidebar.markdown('<div class="settings-label">Confidence</div>', unsafe_allow_html=True)
    confidence = st.sidebar.slider('', min_value=0.0, max_value=1.0, value=0.25, key='confidence')
    
    st.sidebar.markdown('<div class="settings-label">Save Image</div>', unsafe_allow_html=True)
    save_img = st.sidebar.checkbox('', key='save_img')
    
    st.sidebar.markdown('<div class="settings-label">Enable GPU</div>', unsafe_allow_html=True)
    enable_GPU = st.sidebar.checkbox('', key='enable_GPU')

    st.sidebar.markdown('<div class="settings-label">Use Custom Classes</div>', unsafe_allow_html=True)
    custom_classes = st.sidebar.checkbox('', key='custom_classes')

    assigned_class_id = []
    names = ['car', 'motorcycle', 'bus', 'train', 'truck', 'traffic light', 'bike']
    if custom_classes:
        st.sidebar.markdown('<div class="settings-label">Select The Custom Classes</div>', unsafe_allow_html=True)
        assigned_class = st.sidebar.multiselect('', names, default='bike', key='assigned_class')
        for each in assigned_class:
            assigned_class_id.append(names.index(each))

    st.sidebar.markdown('<div class="settings-label">Choose media type</div>', unsafe_allow_html=True)
    media_type = st.sidebar.radio("", ('Video', 'Image', 'Webcam'), key='media_type')

    # Load the YOLO models
    license_plate_model = YOLO('D:\\AI_Project\\Advanced_Recognization_of_Nepali_License_Plates_using_YOLOv8_and_OCR_Technologies\\DeepLearning_Models\\best24.pt')
    character_model = YOLO('D:\\AI_Project\\Advanced_Recognization_of_Nepali_License_Plates_using_YOLOv8_and_OCR_Technologies\\DeepLearning_Models\\best.pt')

    def process_image(image):
        results = license_plate_model.predict(image)
        return results

    def display_detected_characters(detected_characters):
        if detected_characters:
            detected_characters.sort(key=lambda x: (x[2], x[1]))

            rows = []
            current_row = []
            previous_y = detected_characters[0][2]

            for char in detected_characters:
                label, x, y = char
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
                english_row = " ".join(label for label, _ in row)
                nepali_row = " ".join(nepali_labels.get(label, label) for label, _ in row)
                
                english_output += f"{english_row}\n"
                nepali_output += f"{nepali_row}\n"

            english_html_output = english_output.strip().replace("\n", "<br>")
            nepali_html_output = nepali_output.strip().replace("\n", "<br>")

            st.sidebar.markdown(f'<div class="number-plate">Number Plate (English):<br>{english_html_output}</div>', unsafe_allow_html=True)
            st.sidebar.markdown(f'<div class="number-plate">Number Plate (Nepali):<br>{nepali_html_output}</div>', unsafe_allow_html=True)

            pradesh = None
            if any("BA" in cls or "Pradesh" in cls or "Bagmati" in cls for cls in [char[0] for char in detected_characters]):
                pradesh = "Bagmati"

            if pradesh:
                st.sidebar.markdown(f'<div class="province">Province: {pradesh} (बागमती)</div>', unsafe_allow_html=True)

    if media_type == 'Image':
        image_file_buffer = st.sidebar.file_uploader("Upload an image", type=["jpg", "jpeg", "png"], key='image_file_buffer')
        if image_file_buffer is not None:
            tffile = tempfile.NamedTemporaryFile(delete=False)
            tffile.write(image_file_buffer.read())
            img = cv2.imread(tffile.name)
            st.sidebar.markdown('<div class="settings-label">Input Image</div>', unsafe_allow_html=True)
            st.sidebar.image(img, use_column_width=True)

            if st.button('Process Image', key='process_image'):
                results = process_image(img)
                license_plate_class_index = 1  # Update this if needed for your model

                detected_characters = []  # To store detected characters and their positions

                for result in results:
                    filtered_boxes = [box for box in result.boxes if int(box.cls.item()) == license_plate_class_index]
                    for box in filtered_boxes:
                        try:
                            cls = int(box.cls.item())
                            conf = float(box.conf.item())
                            label = f"{license_plate_model.names[cls]} {conf:.2f}"
                            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
                            cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

                            license_plate_region = img[y1:y2, x1:x2]
                            zoomed_license_plate = cv2.resize(license_plate_region, (224, 224))  # Adjust size as needed
                            char_results = character_model.predict(zoomed_license_plate)

                            for char_result in char_results:
                                for char_box in char_result.boxes:
                                    try:
                                        char_cls = int(char_box.cls.item())
                                        char_conf = float(char_box.conf.item())
                                        char_label = f"{character_model.names[char_cls]} {char_conf:.2f}"
                                        char_x1, char_y1, char_x2, char_y2 = map(int, char_box.xyxy[0].tolist())

                                        orig_char_x1 = x1 + char_x1 * ((x2 - x1) / 224)
                                        orig_char_y1 = y1 + char_y1 * ((y2 - y1) / 224)
                                        orig_char_x2 = x1 + char_x2 * ((x2 - x1) / 224)
                                        orig_char_y2 = y1 + char_y2 * ((y2 - y1) / 224)

                                        detected_characters.append((character_model.names[char_cls], orig_char_x1, orig_char_y1))

                                        cv2.rectangle(img, (int(orig_char_x1), int(orig_char_y1)), (int(orig_char_x2), int(orig_char_y2)), (0, 255, 0), 2)
                                        cv2.putText(img, char_label, (int(orig_char_x1), int(orig_char_y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                                    except Exception as e:
                                        st.write(f"Error processing char box: {e}")
                                        st.write(f"Char box data: {char_box}")

                        except Exception as e:
                            st.write(f"Error processing box: {e}")
                            st.write(f"Box data: {box}")

                detected_characters.sort(key=lambda x: x[1])

                st.markdown('<div class="settings-label">Processed Image</div>', unsafe_allow_html=True)
                st.image(img, use_column_width=True)

                display_detected_characters(detected_characters)

    elif media_type == 'Webcam':
        st.sidebar.markdown('<div class="settings-label">Webcam Capture</div>', unsafe_allow_html=True)
        webcam_image = st.camera_input("Capture an image from the webcam")

        if webcam_image is not None:
            img = cv2.imdecode(np.frombuffer(webcam_image.read(), np.uint8), 1)
            st.sidebar.markdown('<div class="settings-label">Captured Image</div>', unsafe_allow_html=True)
            st.sidebar.image(img, use_column_width=True)

            if st.button('Process Image from Webcam', key='process_webcam_image'):
                results = process_image(img)
                license_plate_class_index = 1  # Update this if needed for your model

                detected_characters = []  # To store detected characters and their positions

                for result in results:
                    filtered_boxes = [box for box in result.boxes if int(box.cls.item()) == license_plate_class_index]
                    for box in filtered_boxes:
                        try:
                            cls = int(box.cls.item())
                            conf = float(box.conf.item())
                            label = f"{license_plate_model.names[cls]} {conf:.2f}"
                            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
                            cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

                            license_plate_region = img[y1:y2, x1:x2]
                            zoomed_license_plate = cv2.resize(license_plate_region, (224, 224))  # Adjust size as needed
                            char_results = character_model.predict(zoomed_license_plate)

                            for char_result in char_results:
                                for char_box in char_result.boxes:
                                    try:
                                        char_cls = int(char_box.cls.item())
                                        char_conf = float(char_box.conf.item())
                                        char_label = f"{character_model.names[char_cls]} {char_conf:.2f}"
                                        char_x1, char_y1, char_x2, char_y2 = map(int, char_box.xyxy[0].tolist())

                                        orig_char_x1 = x1 + char_x1 * ((x2 - x1) / 224)
                                        orig_char_y1 = y1 + char_y1 * ((y2 - y1) / 224)
                                        orig_char_x2 = x1 + char_x2 * ((x2 - x1) / 224)
                                        orig_char_y2 = y1 + char_y2 * ((y2 - y1) / 224)

                                        detected_characters.append((character_model.names[char_cls], orig_char_x1, orig_char_y1))

                                        cv2.rectangle(img, (int(orig_char_x1), int(orig_char_y1)), (int(orig_char_x2), int(orig_char_y2)), (0, 255, 0), 2)
                                        cv2.putText(img, char_label, (int(orig_char_x1), int(orig_char_y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                                    except Exception as e:
                                        st.write(f"Error processing char box: {e}")
                                        st.write(f"Char box data: {char_box}")

                        except Exception as e:
                            st.write(f"Error processing box: {e}")
                            st.write(f"Box data: {box}")

                detected_characters.sort(key=lambda x: x[1])

                st.markdown('<div class="settings-label">Processed Image from Webcam</div>', unsafe_allow_html=True)
                st.image(img, use_column_width=True)

                display_detected_characters(detected_characters)

    elif media_type == 'Video':
        video_file_buffer = st.sidebar.file_uploader("Upload a video", type=["mp4", "avi", "mov"], key='video_file_buffer')
        if video_file_buffer is not None:
            tffile = tempfile.NamedTemporaryFile(delete=False)
            tffile.write(video_file_buffer.read())
            cap = cv2.VideoCapture(tffile.name)

            ret, frame = cap.read()  # Read the first frame
            if ret:
                st.sidebar.markdown('<div class="settings-label">Extracted Frame</div>', unsafe_allow_html=True)
                st.sidebar.image(frame, use_column_width=True)

                if st.button('Process Frame', key='process_frame'):
                    results = process_image(frame)
                    license_plate_class_index = 1  # Update this if needed for your model

                    detected_characters = []  # To store detected characters and their positions

                    for result in results:
                        filtered_boxes = [box for box in result.boxes if int(box.cls.item()) == license_plate_class_index]
                        for box in filtered_boxes:
                            try:
                                cls = int(box.cls.item())
                                conf = float(box.conf.item())
                                label = f"{license_plate_model.names[cls]} {conf:.2f}"
                                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

                                license_plate_region = frame[y1:y2, x1:x2]
                                zoomed_license_plate = cv2.resize(license_plate_region, (224, 224))  # Adjust size as needed
                                char_results = character_model.predict(zoomed_license_plate)

                                for char_result in char_results:
                                    for char_box in char_result.boxes:
                                        try:
                                            char_cls = int(char_box.cls.item())
                                            char_conf = float(char_box.conf.item())
                                            char_label = f"{character_model.names[char_cls]} {char_conf:.2f}"
                                            char_x1, char_y1, char_x2, char_y2 = map(int, char_box.xyxy[0].tolist())

                                            orig_char_x1 = x1 + char_x1 * ((x2 - x1) / 224)
                                            orig_char_y1 = y1 + char_y1 * ((y2 - y1) / 224)
                                            orig_char_x2 = x1 + char_x2 * ((x2 - x1) / 224)
                                            orig_char_y2 = y1 + char_y2 * ((y2 - y1) / 224)

                                            detected_characters.append((character_model.names[char_cls], orig_char_x1, orig_char_y1))

                                            cv2.rectangle(frame, (int(orig_char_x1), int(orig_char_y1)), (int(orig_char_x2), int(orig_char_y2)), (0, 255, 0), 2)
                                            cv2.putText(frame, char_label, (int(orig_char_x1), int(orig_char_y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                                        except Exception as e:
                                            st.write(f"Error processing char box: {e}")
                                            st.write(f"Char box data: {char_box}")

                            except Exception as e:
                                st.write(f"Error processing box: {e}")
                                st.write(f"Box data: {box}")

                    detected_characters.sort(key=lambda x: x[1])

                    st.markdown('<div class="settings-label">Processed Frame</div>', unsafe_allow_html=True)
                    st.image(frame, use_column_width=True)

                    display_detected_characters(detected_characters)

    st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
