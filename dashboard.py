import streamlit as st
from ultralytics import YOLO
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import cv2

# ==========================
# Load Models
# ==========================
@st.cache_resource
def load_models():
    # Path disesuaikan dengan struktur folder
    yolo_model = YOLO("model/Yumnaa Alifah_Laporan 4.pt")  # Model deteksi objek
    classifier = tf.keras.models.load_model("model/classifier_model.h5")  # Model klasifikasi
    return yolo_model, classifier


# ==========================
# Main App
# ==========================
def main():
    st.title("Dashboard Deteksi dan Klasifikasi Gambar")

    # Load model sekali saja
    yolo_model, classifier = load_models()

    # Upload gambar
    uploaded_file = st.file_uploader("Upload gambar untuk deteksi:", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Buka gambar
        img = Image.open(uploaded_file)
        st.image(img, caption="Gambar yang diunggah", use_column_width=True)

        # Simpan sementara untuk YOLO
        img_path = "temp_image.jpg"
        img.save(img_path)

        # Deteksi objek dengan YOLO
        results = yolo_model(img_path)
        annotated_frame = results[0].plot()

        # Tampilkan hasil deteksi
        st.image(annotated_frame, caption="Hasil Deteksi YOLO", use_column_width=True)

        # Proses klasifikasi
        img_for_class = image.load_img(img_path, target_size=(224, 224))  # Sesuaikan ukuran input
        img_array = image.img_to_array(img_for_class)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.0

        # Prediksi
        predictions = classifier.predict(img_array)
        predicted_class = np.argmax(predictions, axis=1)[0]

        st.write("### Hasil Klasifikasi:")
        st.write(f"Label Prediksi: **{predicted_class}**")

if __name__ == "__main__":
    main()
