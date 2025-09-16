import streamlit as st
import pandas as pd
import joblib
from PIL import Image

# Loading The Model
# I used a try-except block to handle the case where the file might not exist.
try:
    model = joblib.load('fruit_pipeline.joblib')
except FileNotFoundError:
    st.error("Model file not found. Please run `model_training.py` first to generate it.")
    st.stop() # Stop the app from running further

# App Configuration
st.set_page_config(
    page_title="Fruit Classifier",
    page_icon="🍎",
    layout="centered"
)


# UI Setup
st.title('🍌 Fruit Classifier 🍇')
st.write("Fill in the details below to get a prediction of the fruit type.")


# Mapping fruit names to their image files and a background color
FRUIT_PROPS = {
    "apple": {"image": "assets/apple.png", "color": "#ffcccc"},
    "banana": {"image": "assets/banana.png", "color": "#fff0b3"},
    "grape": {"image": "assets/grape.png", "color": "#e6ccff"},
}


# Creating input fields in the sidebar
st.sidebar.header('Input Features')

# Defining the options for the dropdowns
color_options = ['Yellow', 'Pink', 'Pale Yellow', 'Creamy White', 'Green', 'Red']
size_options = ['Tiny', 'Small', 'Medium', 'Large']

# Creating the input widgets
input_color = st.sidebar.selectbox('Color', options=color_options)
input_size = st.sidebar.selectbox('Size', options=size_options)
input_weight = st.sidebar.number_input('Weight (grams)', min_value=1.0, max_value=200.0, value=80.0, step=1.0)


# Prediction Logic
# Created a button to trigger the prediction
if st.sidebar.button('Predict'):
    # Creating a DataFrame from the user's input
    # The model expects a DataFrame with the same column names as the training data
    input_data = pd.DataFrame({
        'color': [input_color],
        'size': [input_size],
        'weight': [input_weight]
    })
    
    st.write("---") # Separator
    st.subheader("Prediction")

    # Using the loaded model to make a prediction
    prediction = model.predict(input_data)[0]
    prediction_proba = model.predict_proba(input_data)
    
    # Get the confidence score for the predicted class
    confidence = prediction_proba.max()

    # Display the result
    fruit_info = FRUIT_PROPS.get(prediction, {"image": None, "color": "#f0f2f6"})
    
    # Use columns for a nice layout
    col1, col2 = st.columns([1, 2])
    
    with col1:
        if fruit_info["image"] and Image.open(fruit_info["image"]):
             # Opened the image using Pillow to handle potential errors
            try:
                img = Image.open(fruit_info["image"])
                st.image(img, width=120)
            except FileNotFoundError:
                st.warning(f"Image for {prediction} not found in assets folder.")
        
    with col2:
        # Custom styled output using markdown
        st.markdown(f"""
        <div style="background-color:{fruit_info['color']}; padding: 20px; border-radius: 10px;">
            <h3>The fruit is: <strong>{prediction.capitalize()}</strong></h3>
            <p><strong>Confidence:</strong> {confidence:.2%}</p>
        </div>
        """, unsafe_allow_html=True)