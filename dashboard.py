import streamlit as st
from ultralytics import YOLO
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import cv2
import os

# ==========================
# Load Models
# ==========================
@st.cache_resource
def load_models():
    # Pastikan nama file model tidak mengandung spasi
    yolo_path = "model/Yumnaa_Alifah_Laporan_4.pt"
    classifier_path = "model/classifier_model.h5"

    # Cek keberadaan file
    if not os.path.exists(yolo_path):
        st.error(f"❌ File YOLO tidak ditemukan di {yolo_path}")
        st.stop()
    if not os.path.exists(classifier_path):
        st.error(f"❌ File classifier tidak ditemukan di {classifier_path}")
        st.stop()

    # Load model
    yolo_model = YOLO(yolo_path)
    classifier = tf.keras.models.load_model(classifier_path)
    return yolo_model, classifier


# ==========================
# Deteksi Objek
# ==========================
def detect_objects(yolo_model, img):
    results = yolo_model.predict(img)
    if len(results[0].boxes) == 0:
        return img, False
    else:
        result_img = results[0].plot()
        return result_img, True


# ==========================
# Prediksi Klasifikasi
# ==========================
def classify_image(classifier, img):
    img_resized = img.resize((224, 224))  # sesuaikan ukuran input model
    img_array = image.img_to_array(img_resized)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    prediction = classifier.predict(img_array)
    predicted_class = np.argmax(prediction, axis=1)[0]
    confidence = float(np.max(prediction))
    return predicted_class, confidence


# ==========================
# Streamlit UI
# ==========================
def main():
    st.title("🚗 Hasil Deteksi Objek dan Klasifikasi Botol")

    # Load model
    yolo_model, classifier = load_models()

    uploaded_file = st.file_uploader("📤 Upload gambar botol", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        img = Image.open(uploaded_file).convert("RGB")
        st.image(img, caption="🖼️ Gambar Asli", width=300)

        # Deteksi objek dengan YOLO
        result_img, detected = detect_objects(yolo_model, img)

        if detected:
            st.image(result_img, caption="✅ Hasil Deteksi YOLO", width=300)

            # Klasifikasi
            predicted_class, confidence = classify_image(classifier, img)
            st.success(f"🎯 Kelas Prediksi: {predicted_class} (Confidence: {confidence:.2f})")
        else:
            st.warning("⚠️ Tidak ada objek terdeteksi. Pastikan model YOLO sudah dilatih untuk mendeteksi botol.")

# ==========================
# Jalankan Aplikasi
# ==========================
if __name__ == "__main__":
    main()
