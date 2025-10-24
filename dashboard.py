import streamlit as st
from ultralytics import YOLO
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import os

# ==========================
# Fungsi Load Models
# ==========================
@st.cache_resource(show_spinner="🔄 Memuat model YOLO dan classifier...")
def load_models():
    yolo_path = "model/Yumnaa_Alifah_Laporan_4.pt"
    classifier_path = "model/classifier_model.h5"

    # Cek file model
    if not os.path.exists(yolo_path):
        raise FileNotFoundError(f"Model YOLO tidak ditemukan di: {yolo_path}")
    if not os.path.exists(classifier_path):
        raise FileNotFoundError(f"Model classifier tidak ditemukan di: {classifier_path}")

    # Load model YOLO dan classifier
    yolo_model = YOLO(yolo_path)
    classifier = tf.keras.models.load_model(classifier_path)
    return yolo_model, classifier


# ==========================
# Deteksi Objek YOLO
# ==========================
def detect_objects(yolo_model, img):
    results = yolo_model.predict(img, verbose=False)
    if results and len(results[0].boxes) > 0:
        result_img = results[0].plot()
        return result_img, True
    else:
        return img, False


# ==========================
# Klasifikasi Gambar
# ==========================
def classify_image(classifier, img):
    img_resized = img.resize((224, 224))
    img_array = image.img_to_array(img_resized)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    prediction = classifier.predict(img_array)
    predicted_class = np.argmax(prediction, axis=1)[0]
    confidence = float(np.max(prediction))
    return predicted_class, confidence


# ==========================
# Tampilan Streamlit
# ==========================
def main():
    st.set_page_config(page_title="Deteksi & Klasifikasi", page_icon="🤖", layout="centered")

    st.title("🤖 Dashboard Deteksi & Klasifikasi Gambar")
    st.write("Aplikasi ini menggunakan **YOLOv8** untuk deteksi objek dan **CNN Classifier (H5)** untuk klasifikasi gambar.")

    menu = st.sidebar.radio("Pilih Mode:", ["📦 Deteksi Objek (YOLO)", "🧠 Klasifikasi Gambar"])

    try:
        yolo_model, classifier = load_models()
    except Exception as e:
        st.error(f"Gagal memuat model: {e}")
        st.stop()

    uploaded_file = st.file_uploader("📤 Upload Gambar", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        img = Image.open(uploaded_file).convert("RGB")
        st.image(img, caption="🖼️ Gambar Asli", use_container_width=True)

        if menu == "📦 Deteksi Objek (YOLO)":
            st.info("🔍 Mendeteksi objek menggunakan model YOLO...")
            result_img, detected = detect_objects(yolo_model, img)

            if detected:
                st.image(result_img, caption="✅ Hasil Deteksi YOLO", use_container_width=True)
            else:
                st.warning("⚠️ Tidak ada objek terdeteksi dalam gambar ini.")

        elif menu == "🧠 Klasifikasi Gambar":
            st.info("📊 Mengklasifikasi gambar menggunakan CNN...")
            predicted_class, confidence = classify_image(classifier, img)
            label_map = {
                0: "Plastik",
                1: "Kertas",
                2: "Logam",
                3: "Lainnya"
            }
            kelas = label_map.get(predicted_class, f"Class {predicted_class}")
            st.success(f"🎯 Hasil Prediksi: **{kelas}** (Confidence: {confidence:.2f})")


if __name__ == "__main__":
    main()
