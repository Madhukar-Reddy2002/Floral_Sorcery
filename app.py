import streamlit as st
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import tensorflow as tf
import matplotlib.pyplot as plt
import base64

# Load the pre-trained model
model_path = "./MyModel5.h5"
model = tf.keras.models.load_model(model_path)

CLASS_NAMES = ['balloon flower', 'black-eyed susan', 'foxglove', 'frangipani', 'jasmine', 'lotus lotus', 'orange hibiscus', 'orange marigold', 'oxeye daisy', 'pink hibiscus', 'pink rose', 'red hibiscus', 'redRose', 'stemless gentian', 'sunflower', 'thorn apple', 'water lily', 'yellow hibiscus', 'yellow marigold', 'yellow rose']

# Set the theme for the app
st.set_page_config(page_title="Floral Sorcery", page_icon="🌸", layout="wide", initial_sidebar_state="collapsed")

# Load the background image
background_image_path = "./dp.jpg"

# Create a text overlay
background_image = Image.open(background_image_path)
draw = ImageDraw.Draw(background_image)
text = "Floral Sorcery"
x = (background_image.width )
y = 50
draw.text((x, y), text, fill=(255, 255, 255))

# Encode the background image as base64
background_image_bytes = background_image.tobytes()
background_data = base64.b64encode(background_image_bytes).decode()

# Set the background for the app
st.markdown(
   f"""
   <style>
   .stApp {{
       background-image: url("data:image/png;base64,{background_data}");
       background-size: cover;
   }}
   </style>
   """,
   unsafe_allow_html=True,
)

# Page Title and Introduction
st.title("🌺 Floral Sorcery 🌼")
st.markdown(
   """
   <p style="font-family: 'Times New Roman', serif; font-size: 20px; color: #fff9e6; text-shadow: 2px 2px 4px rgba(0,0,0,0.5);">
   Welcome to the enchanted realm of Floral Sorcery, where the whispers of nature's most exquisite creations come alive through the arcane arts of artificial intelligence. 🌸✨
   
   Here, every petal holds a secret, and every blossom tells a tale as old as time itself. Unveil the true essence of flowers with our mystical image classifier, and let their beauty captivate your senses in ways you've never imagined. 🌺🌹🌼
   </p>
   """,
   unsafe_allow_html=True,
)

# Upload file and make predictions
uploaded_file = st.file_uploader("🌻 Choose a Flower Image to Unravel Its Secrets...", type=["jpg", "jpeg"])

if uploaded_file is not None:
   # Display the uploaded image with flair
   image = Image.open(uploaded_file)
   image_preview = image.resize((250, 250))
   st.image(image_preview, caption="🖼️ Image Preview", use_column_width=True)
   
   # Preprocess the image
   image = image.resize((250, 250))
   image_array = np.array(image)
   image_batch = np.expand_dims(image_array, axis=0)

   # Make a prediction
   prediction = model.predict(image_batch)
   top_class_index = np.argmax(prediction[0])
   top_class_name = CLASS_NAMES[top_class_index]
   top_confidence = float(prediction[0][top_class_index])

   # Display the top prediction with splendor
   st.subheader("🌸 The Floral Incantation 🌸")
   st.markdown(
       f"""
       <p style="font-family: 'Times New Roman', serif; font-size: 24px; color: #fff9e6; text-shadow: 2px 2px 4px rgba(0,0,0,0.5);">
       Through the mystical lens of our artificial sorcery, we unveil the true nature of this enchanted blossom – **'{top_class_name}'**! With a confidence of {top_confidence:.2%}, let its beauty and charm captivate your senses, as if plucked straight from the gardens of the fae! ✨
       </p>
       """,
       unsafe_allow_html=True,
   )

   # Display the top 3 predictions on the enchanting graph
   top_classes = np.argsort(prediction[0])[::-1][:3]  # Get indices of top 3 classes
   top_classes_names = [CLASS_NAMES[i] for i in top_classes]
   top_confidences = [float(prediction[0][i]) for i in top_classes]

   # Create a bar chart for top predictions with ethereal colors
   fig, ax = plt.subplots(figsize=(12, 6))
   ax.bar(top_classes_names, top_confidences, color=['#FFC300', '#FF5733', '#C70039'])
   ax.set_ylabel('Confidence', fontweight='bold', color='#fff9e6', fontfamily="Times New Roman, serif")
   ax.set_title('🌟 The Floral Oracle: Top Predictions of the Flower Realm 🌟', fontsize=24, fontweight='bold', color='#fff9e6', fontfamily="Times New Roman, serif")
   ax.tick_params(colors='#fff9e6', labelsize=14)
   fig.patch.set_facecolor('#2d2d2d')
   ax.spines['bottom'].set_color('#fff9e6')
   ax.spines['top'].set_color('#fff9e6')
   ax.spines['right'].set_color('#fff9e6')
   ax.spines['left'].set_color('#fff9e6')
   ax.set_facecolor('#2d2d2d')
   st.pyplot(fig)