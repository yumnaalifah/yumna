import streamlit as st
from ultralytics import YOLO
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import cv2
import time

# ==========================
# Konfigurasi Dashboard
# ==========================
st.set_page_config(
    page_title="Intelligent Vision Dashboard",
    page_icon="🧠",
    layout="wide",
)

# ==========================
# Load Models
# ==========================
@st.cache_resource
def load_models():
    yolo_model = YOLO("model/Yumnaa Alifah_Laporan 4.pt")  # YOLO untuk objek
    classifier = tf.keras.models.load_model("model/classifier_model.h5")  # CNN untuk sampah
    return yolo_model, classifier

yolo_model, classifier = load_models()

# ==========================
# UI Header
# ==========================
st.markdown(
    """
    <style>
    .big-font {
        font-size:35px !important;
        font-weight:600;
        text-align:center;
        color:#3A3B3C;
    }
    .subtitle {
        text-align:center;
        color:gray;
        font-size:18px;
        margin-bottom:30px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<p class="big-font">🧠 Intelligent Vision Dashboard</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Deteksi Objek (YOLO) & Klasifikasi Sampah (CNN)</p>', unsafe_allow_html=True)

# ==========================
# Sidebar Menu
# ==========================
menu = st.sidebar.radio("🔍 Pilih Mode:", ["Deteksi Objek (YOLO)", "Klasifikasi Sampah (CNN)"])
st.sidebar.markdown("---")
st.sidebar.info("📁 Pastikan file gambar berformat JPG, JPEG, atau PNG.")

# ==========================
# Upload File
# ==========================
uploaded_file = st.file_uploader("Unggah Gambar di sini", type=["jpg", "jpeg", "png"])

if uploaded_file:
    col1, col2 = st.columns(2)
    with col1:
        img = Image.open(uploaded_file)
        st.image(img, caption="📸 Gambar Asli", use_container_width=True)

    with col2:
        if menu == "Deteksi Objek (YOLO)":
            st.markdown("### 🚘 Hasil Deteksi Objek")
            with st.spinner("Mendeteksi objek..."):
                results = yolo_model(img)
                result_img = results[0].plot()
                time.sleep(1)
            st.image(result_img, caption="Hasil Deteksi YOLO", use_container_width=True)

            # Menampilkan hasil deteksi
            detected_objects = [r.names[int(cls)] for cls in results[0].boxes.cls]
            if detected_objects:
                st.success("✅ Objek Terdeteksi:")
                st.write(", ".join(set(detected_objects)))
            else:
                st.warning("Tidak ada objek terdeteksi.")

        elif menu == "Klasifikasi Sampah (CNN)":
            st.markdown("### 🗑️ Hasil Klasifikasi Sampah")
            with st.spinner("Mengklasifikasi gambar..."):
                img_resized = img.resize((224, 224))
                img_array = image.img_to_array(img_resized)
                img_array = np.expand_dims(img_array, axis=0) / 255.0

                prediction = classifier.predict(img_array)
                class_index = np.argmax(prediction)
                prob = np.max(prediction)

                # Label kelas (ubah sesuai label model kamu)
                labels = ["Sampah Plastik", "Sampah Kertas", "Sampah Logam", "Sampah Kaca", "Sampah Organik"]

                time.sleep(1)
            st.success(f"✅ Jenis Sampah: **{labels[class_index]}**")
            st.progress(float(prob))
            st.write(f"**Probabilitas:** {prob:.2%}")

            # Penjelasan tambahan
            if labels[class_index] == "Sampah Plastik":
                st.info("♻️ Plastik sulit terurai, sebaiknya didaur ulang menjadi produk baru.")
            elif labels[class_index] == "Sampah Kertas":
                st.info("📄 Kertas dapat didaur ulang untuk mengurangi penebangan pohon.")
            elif labels[class_index] == "Sampah Logam":
                st.info("🔩 Logam bisa dilebur kembali menjadi bahan baku industri.")
            elif labels[class_index] == "Sampah Kaca":
                st.info("🧴 Kaca dapat digunakan kembali atau dilebur menjadi bentuk baru.")
            else:
                st.info("🌱 Sampah organik bisa diolah menjadi kompos alami.")

else:
    st.info("⬆️ Silakan unggah gambar terlebih dahulu.")

# ==========================
# Footer
# ==========================
st.markdown("---")
st.markdown(
    """
    <p style="text-align:center; color:gray;">
    Dibuat oleh <b>Yumnaa Alifah</b> untuk Proyek UAS 💡 | 2025
    </p>
    """,
    unsafe_allow_html=True,
)

