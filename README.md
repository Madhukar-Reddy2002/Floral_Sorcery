# Floral Sorcery: A Streamlit Flower Classification App

Embrace the magic of flowers with Floral Sorcery, a captivating Streamlit app that unveils the mysteries hidden within their blooms!

## Key Features:

- **Image Upload:** Upload flower images in JPG or JPEG format.
- **Real-time Predictions:** Employ a pre-trained TensorFlow model to predict the flower type and its confidence score.
- **Engaging Visuals:** Witness your uploaded image displayed alongside the predicted flower and its confidence, presented in a captivating manner.
- **Enriched Knowledge:** Delve deeper with additional information about the predicted flower, including uses, characteristics, and growing conditions (if available in the knowledge base).
- **Top Predictions Chart:** Visualize the top 3 most likely flower classes with a beautifully styled bar chart.
- **Immersive Experience:** The app's design evokes a mystical and enchanting theme, enhancing the user experience.

## Requirements:

- Python 3.x ([Download Python](https://www.python.org/downloads/))
- Streamlit ([Streamlit GitHub](https://github.com/streamlit/streamlit))
- TensorFlow ([TensorFlow Installation](https://www.tensorflow.org/install/pip))
- NumPy ([NumPy Website](https://numpy.org/))
- Pillow ([Pillow Documentation](https://pillow.readthedocs.io/))

## Instructions:

1. Ensure you have the necessary libraries installed (`pip install streamlit tensorflow numpy pillow`).
2. Replace `./MyModel.h5` in the code with the path to your pre-trained TensorFlow model file (`model_path`).
3. Adjust the `CLASS_NAMES` list to match the flower classes your model predicts.
4. If you have flower information for additional classes, populate the `FLOWER_INFO` dictionary.
5. Run the app using `streamlit run app.py`.

## User Guide:

1. Visit the app in your web browser.
2. Upload a flower image (JPG or JPEG format).
3. The app will process the image and unveil the predicted flower type, along with its confidence score, in a visually engaging way.
4. If available, you'll also see the flower's uses, characteristics, and growing conditions.
5. Explore the top 3 predictions bar chart to discover other potential flower matches.

## Further Considerations:

- Implement error handling for potential issues during image processing or prediction.
- Explore options for integrating more detailed or interactive flower information panels.
- Consider deploying the app to a cloud platform for wider accessibility.

Embrace the beauty and mysteries of the floral world with Floral Sorcery!
