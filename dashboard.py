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
    yolo_model = YOLO("model/Yumnaa Alifah_Laporan 4.pt")  # Model deteksi objek
    classifier = tf.keras.models.load_model("modelclassifier_model.h5")  # Model klasifikasi
    return yolo_model, classifier


# ==========================
# Main App
# ==========================
st.title("🚀 Aplikasi Deteksi & Klasifikasi Citra")

yolo_model, classifier = load_models()

uploaded_file = st.file_uploader("Unggah gambar untuk deteksi dan klasifikasi", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption="Gambar diunggah", use_container_width=True)

    # Simpan gambar sementara
    img_path = "temp.jpg"
    img.save(img_path)

    # Deteksi objek dengan YOLO
    results = yolo_model(img_path)
    result_img = results[0].plot()  # Hasil deteksi berupa array gambar

    st.image(result_img, caption="Hasil Deteksi YOLO", use_container_width=True)

    # ==========================
    # Klasifikasi dengan model .h5
    # ==========================
    img_resized = img.resize((224, 224))  # ubah ukuran sesuai model klasifikasi
    img_array = image.img_to_array(img_resized)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    prediction = classifier.predict(img_array)
    predicted_class = np.argmax(prediction, axis=1)

    st.success(f"✅ Kelas Prediksi: {predicted_class[0]}")
