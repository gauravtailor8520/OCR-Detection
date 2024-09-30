import os
import streamlit as st
import cv2
import pytesseract
from PIL import Image
import numpy as np
from transformers import AutoModel



# Set the path for the Tesseract executable
pytesseract.pytesseract.tesseract_cmd = r'https://github.com/gauravtailor8520/Parimal/blob/main/tesseract.exe'

# Set the TESSDATA_PREFIX environment variable
os.environ['TESSDATA_PREFIX'] = r'https://github.com/gauravtailor8520/Parimal/tree/main/tessdata' 


# Function to preprocess the image
def preprocess_image(image):
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply thresholding
    _, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Denoise the image
    denoised = cv2.fastNlMeansDenoising(binary, None, 10, 7, 21)
    
    return denoised

# Function to extract text from the image
def extract_text_from_image(image_path):
    # Load the image using OpenCV
    image = cv2.imread(image_path)
    
    # Check if the image was loaded successfully
    if image is None:
        raise ValueError(f"Image not found or could not be loaded at path: {image_path}")
    
    # Preprocess the image
    preprocessed_image = preprocess_image(image)
    
    # Use pytesseract to do OCR on the image for both Hindi and English
    extracted_text = pytesseract.image_to_string(preprocessed_image, lang='hin+eng')
    
    return extracted_text

# Streamlit app layout
st.title("OCR with Tesseract")

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Read and display image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Save the uploaded file temporarily
    image_path = "uploaded_image.png"
    image.save(image_path)

    # Extract text from the uploaded image
    if st.button("Extract Text"):
        try:
            extracted_text = extract_text_from_image(image_path)
            st.success("Text extraction completed!")

            # Display the extracted text
            st.subheader("Extracted Text:")
            st.write(extracted_text)
        except Exception as e:
            st.error(f"Error extracting text: {e}")
