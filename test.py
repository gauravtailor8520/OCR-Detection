import os
import streamlit as st
import cv2
import pytesseract
from PIL import Image
import numpy as np

# Define global paths for Tesseract executable and tessdata directory
BASE_PATH = os.path.dirname(os.path.abspath(__file__))  # Get the current file's directory
TESSERACT_PATH = os.path.join(BASE_PATH, 'tesseract.exe')  # Path to the tesseract executable
TESSDATA_PATH = os.path.join(BASE_PATH, 'tessdata')  # Path to the tessdata directory

# Set the path for the Tesseract executable
pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH

# Set the TESSDATA_PREFIX environment variable to the tessdata directory
os.environ['TESSDATA_PREFIX'] = TESSDATA_PATH

# Debugging output
st.write(f"Tesseract Path: {TESSERACT_PATH}")
st.write(f"Tessdata Path: {TESSDATA_PATH}")

# Function to preprocess the image
def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    denoised = cv2.fastNlMeansDenoising(binary, None, 10, 7, 21)
    return denoised

# Function to extract text from the image
def extract_text_from_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Image not found or could not be loaded at path: {image_path}")
    preprocessed_image = preprocess_image(image)
    
    # Use pytesseract to do OCR on the image for both Hindi and English
    extracted_text = pytesseract.image_to_string(preprocessed_image, lang='hin+eng')
    return extracted_text

# Streamlit app layout
st.title("OCR with Tesseract")

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    image_path = "uploaded_image.png"
    image.save(image_path)

    if st.button("Extract Text"):
        try:
            extracted_text = extract_text_from_image(image_path)
            st.success("Text extraction completed!")
            st.subheader("Extracted Text:")
            st.write(extracted_text)
        except Exception as e:
            st.error(f"Error extracting text: {e}")
