import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import matplotlib.pyplot as plt

# Load the pre-trained model
model_path = "./MyModel5.h5"
model = tf.keras.models.load_model(model_path)

CLASS_NAMES = ['balloon flower', 'black-eyed susan', 'foxglove', 'frangipani', 'jasmine', 'lotus lotus', 'orange hibiscus', 'orange marigold', 'oxeye daisy', 'pink hibiscus', 'pink rose', 'red hibiscus', 'redRose', 'stemless gentian', 'sunflower', 'thorn apple', 'water lily', 'yellow hibiscus', 'yellow marigold', 'yellow rose']

# Set the theme for the app
st.set_page_config(page_title="Floral Sorcery", page_icon="🌸", layout="wide", initial_sidebar_state="collapsed")

# Page Title
st.title("🌺 Floral Sorcery 🌼")
st.markdown(
   """
   <p style="font-family: 'Times New Roman', serif; font-size: 20px; color: #fff9e6; text-shadow: 2px 2px 4px rgba(0,0,0,0.5);">
   Here, every petal holds a secret, and every blossom tells a tale as old as time itself. Unveil the true essence of flowers with our mystical image classifier, and let their beauty captivate your senses in ways you've never imagined. 🌺🌹🌼
   </p>
   """,
   unsafe_allow_html=True,
)

# Upload file and make predictions
uploaded_file = st.file_uploader("🌻 Choose a Flower Image to Unravel Its Secrets...", type=["jpg", "jpeg"])

if uploaded_file is not None:
    # Display the uploaded image
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

    # Display the top prediction with flair
    st.subheader("🌸 The Floral Incantation 🌸")

    flower_name = f"'{top_class_name}'"  # Enclose class name in quotes for emphasis

    # Define a custom HTML template with styling
    html_template = f"""
    <p style="font-size: 28px; color: #fff9e6; text-shadow: 2px 2px 4px rgba(0,0,0,0.5);">
    Our mystical lens unveils the true nature of the enchanted blossom...  
    <br>  
    Behold, the magnificent <span style="color: #FFC300; font-weight: bold; font-size: 32px;">{flower_name}</span>! 
    <br>  
    With a confidence of {top_confidence:.2%}, its beauty transcends the veil, captivating your senses! ✨
    </p>
    """

    # Display the formatted HTML content 
    st.markdown(html_template, unsafe_allow_html=True)

    # Display the top 3 predictions on a bar chart
    top_classes = np.argsort(prediction[0])[::-1][:3]  # Get indices of top 3 classes
    top_classes_names = [CLASS_NAMES[i] for i in top_classes]
    top_confidences = [float(prediction[0][i]) for i in top_classes]

    # Create a bar chart for top predictions with ethereal colors
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(top_classes_names, top_confidences, color=['#FFC300', '#FF5733', '#C70039'])
    ax.set_ylabel('Confidence', fontweight='bold', color='#fff9e6')
    ax.set_title('🌟 The Floral Oracle: Top Predictions of the Flower Realm 🌟', fontsize=24, fontweight='bold', color='#fff9e6', fontfamily="Times New Roman, serif")
    ax.tick_params(colors='#fff9e6', labelsize=14)
    fig.patch.set_facecolor('#2d2d2d')
    ax.spines['bottom'].set_color('#fff9e6')
    ax.spines['top'].set_color('#fff9e6')
    ax.spines['right'].set_color('#fff9e6')
    ax.spines['left'].set_color('#fff9e6')
    ax.set_facecolor('#2d2d2d')
    st.pyplot(fig)