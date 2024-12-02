import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import io

# Load the trained model
model = load_model('cat_dog_classifier.h5')

# Set page config
st.set_page_config(page_title="Cat vs Dog Classifier", layout="wide")

# Title
st.title("🐱 Cat vs Dog Classifier 🐶")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

def preprocess_image(img):
    # Resize image
    img = img.resize((150, 150))
    # Convert to array
    img_array = image.img_to_array(img)
    # Expand dimensions
    img_array = np.expand_dims(img_array, axis=0)
    # Rescale
    img_array /= 255.
    return img_array

if uploaded_file is not None:
    # Display the uploaded image
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Uploaded Image")
        image_display = Image.open(uploaded_file)
        st.image(image_display, width=300)

    # Make prediction
    try:
        img = Image.open(uploaded_file)
        processed_img = preprocess_image(img)
        prediction = model.predict(processed_img)

        with col2:
            st.subheader("Prediction")
            probability = prediction[0][0]

            if probability > 0.5:
                st.write(f"This is a Dog! 🐶 ({probability:.2%} confidence)")
            else:
                st.write(f"This is a Cat! 🐱 ({(1-probability):.2%} confidence)")

            # Display probability bar
            st.progress(float(probability))
            st.write("Dog probability: ", f"{probability:.2%}")
            st.write("Cat probability: ", f"{(1-probability):.2%}")

    except Exception as e:
        st.error(f"Error processing image: {str(e)}")

# Add instructions
st.markdown("""
### Instructions:
1. Upload an image of a cat or dog using the file uploader above
2. The model will predict whether it's a cat or dog
3. The confidence score shows how sure the model is about its prediction

### Notes:
- Supported formats: JPG, JPEG, PNG
- For best results, use clear images with a single cat or dog
- The model works best with front-facing, well-lit photos
""")
