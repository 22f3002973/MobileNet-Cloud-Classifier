import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image

st.set_page_config(page_title="MobileNetV2 Cloud Classifier")

def main():
    st.title("MobileNetV2 Image Classification")
    st.write("Upload one or more images and the model will classify them.")

    uploaded_files = st.file_uploader("Upload Images", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

    if uploaded_files:
        model = tf.keras.applications.MobileNetV2(weights="imagenet")

        for uploaded_file in uploaded_files:
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, caption=uploaded_file.name, use_column_width=True)

            img = image.resize((224, 224))
            img_array = np.array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)

            predictions = model.predict(img_array)
            decoded = tf.keras.applications.mobilenet_v2.decode_predictions(predictions, top=3)[0]

            st.write(f"### Predictions for {uploaded_file.name}:")
            for _, label, confidence in decoded:
                st.write(f"- **{label}** ({confidence * 100:.2f}%)")

if __name__ == "__main__":
    main()
