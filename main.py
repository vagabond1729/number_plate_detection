import streamlit as st
import cv2
import numpy as np
import torch
import easyocr

# Global variable
EASY_OCR = easyocr.Reader(['en'])
OCR_TH = 0.2

# Function to recognize license plate using EasyOCR
def recognize_plate_easyocr(img, coords):
    xmin, ymin, xmax, ymax = coords
    nplate = img[int(ymin):int(ymax), int(xmin):int(xmax)]  # Cropping the license plate region

    ocr_result = EASY_OCR.readtext(nplate)
    text = filter_text(nplate, ocr_result)

    if len(text) == 1:
        text = text[0].upper()
    return text

# Function to filter out wrong detections
def filter_text(region, ocr_result):
    rectangle_size = region.shape[0] * region.shape[1]
    plate = []

    for result in ocr_result:
        length = np.sum(np.subtract(result[0][1], result[0][0]))
        height = np.sum(np.subtract(result[0][2], result[0][1]))

        if length * height / rectangle_size > OCR_TH:
            plate.append(result[1])
    return plate

# Streamlit app
def main():
    st.title("License Plate Recognition")
    st.sidebar.title("Settings")

    st.sidebar.header("OCR Threshold")
    ocr_threshold = st.sidebar.slider("Select OCR Threshold", 0.1, 1.0, OCR_TH, step=0.05)

    # Create a button to start/stop the camera feed
    start_camera = st.sidebar.button("Start Camera")

    st.sidebar.text("Press 'q' to exit.")

    cap = None
    frame_placeholder = st.empty()

    while True:
        if start_camera:
            # OpenCV webcam capture
            if cap is None:
                cap = cv2.VideoCapture(0)

            ret, frame = cap.read()
            if not ret:
                break

            # Convert frame to grayscale for OCR
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Detect license plates using EasyOCR
            ocr_result = EASY_OCR.readtext(gray_frame)

            for (bbox, text, prob) in ocr_result:
                (top_left, top_right, bottom_right, bottom_left) = bbox
                coords = (top_left[0], top_left[1], bottom_right[0], bottom_right[1])

                # Draw bounding box around the license plate
                cv2.rectangle(frame, top_left, bottom_right, (0, 255, 0), 2)
                cv2.putText(frame, text, (top_left[0], top_left[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Display the processed frame using st.image
            frame_placeholder.image(frame, channels="BGR", use_column_width=True)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        else:
            if cap is not None:
                cap.release()
                cap = None
            frame_placeholder.empty()

    if cap is not None:
        cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()