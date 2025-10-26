import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt

# Function for MobileNetV2 ImageNet model
def mobilenetv2_imagenet():
    st.title("MobileNetV2 Image Classification")
    st.write("Upload one or more images, and the app will classify them using the MobileNetV2 model (trained on ImageNet).")

    uploaded_files = st.file_uploader("Choose image(s)...", type=["jpg", "png"], accept_multiple_files=True)

    if uploaded_files:
        # Load the MobileNetV2 model
        model = tf.keras.applications.MobileNetV2(weights='imagenet')
        results = []

        for uploaded_file in uploaded_files:
            image = Image.open(uploaded_file).convert('RGB')  # Ensure the image is RGB
            st.image(image, caption=f"Uploaded Image: {uploaded_file.name}", use_container_width=True)

            # Preprocess the image
            img = image.resize((224, 224))
            img_array = np.array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)

            # Make predictions
            predictions = model.predict(img_array)
            decoded_predictions = tf.keras.applications.mobilenet_v2.decode_predictions(predictions, top=3)[0]

            # Display predictions
            st.write(f"### Predictions for {uploaded_file.name}:")
            for imagenet_id, label, score in decoded_predictions:
                st.write(f"- **{label.capitalize()}**: {score * 100:.2f}%")

            # Collect predictions for download
            results.append({
                "Image": uploaded_file.name,
                "Top Prediction": f"{decoded_predictions[0][1]} ({decoded_predictions[0][2] * 100:.2f}%)",
                "Second Prediction": f"{decoded_predictions[1][1]} ({decoded_predictions[1][2] * 100:.2f}%)",
                "Third Prediction": f"{decoded_predictions[2][1]} ({decoded_predictions[2][2] * 100:.2f}%)"
            })

            # Graphical display of confidence scores
            st.write("#### Confidence Scores (Top 3)")
            labels = [label.capitalize() for _, label, _ in decoded_predictions]
            scores = [score * 100 for _, _, score in decoded_predictions]
            fig, ax = plt.subplots()
            ax.bar(labels, scores, color=['blue', 'orange', 'green'])
            ax.set_title("Confidence Scores")
            ax.set_ylabel("Confidence (%)")
            st.pyplot(fig)

        # Download csv
        if results:
            st.download_button(
                label="Download Predictions as CSV",
                data=pd.DataFrame(results).to_csv(index=False),
                file_name="mobilenet_predictions.csv",
                mime="text/csv",
            )


# Welcome page
def welcome_page():
    st.title("Welcome to Image Classification App")
    st.write(
        """
        This app allows you to classify images using 
        - **MobileNetV2**: Pre-trained on ImageNet for generic object classification

        ### Features:
        - Upload single or multiple images.
        - Get real-time predictions with confidence scores.
        - Download results as a CSV file.
        - Visualize confidence scores with bar charts.
        """
    )


# Main function to control the navigation
def main():
    st.sidebar.title("Navigation")
    choice = st.sidebar.radio("Choose an Option", ("Welcome", "MobileNetV2 (ImageNet)"))
    
    if choice == "Welcome":
        welcome_page()
    elif choice == "MobileNetV2 (ImageNet)":
        mobilenetv2_imagenet()
    

if __name__ == "__main__":
    main()
