import streamlit as st
from ultralytics import YOLO
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# ==========================
# Konfigurasi Halaman
# ==========================
st.set_page_config(page_title="AI Image Dashboard", page_icon="🧠", layout="wide")

st.title("🧠 Dashboard Deteksi & Klasifikasi Citra")
st.markdown("""
Aplikasi ini memanfaatkan **YOLOv8** untuk *Object Detection* dan **TensorFlow** untuk *Image Classification*.  
Tersedia dua jenis dataset:
- 📱 **Dataset Elektronik** (Laptop, Mobile, Supercar)  
- 🚮 **Dataset Sampah** (Waste Classification Dataset)
""")

# ==========================
# Load Models
# ==========================
@st.cache_resource
def load_models():
    yolo_model = YOLO("Yumnaa Alifah_Laporan 4.pt")  # Model YOLO
    classifier = tf.keras.models.load_model("classifier_model.h5")  # Model Klasifikasi
    return yolo_model, classifier

yolo_model, classifier = load_models()

# ==========================
# Sidebar Menu
# ==========================
with st.sidebar:
    st.header("⚙️ Pengaturan")
    dataset_option = st.selectbox("Pilih Dataset:", ["Elektronik", "Sampah"])
    mode = st.radio("Pilih Mode:", ["Deteksi Objek (YOLO)", "Klasifikasi Gambar"])
    st.markdown("---")
    st.write("Unggah gambar di bawah ini untuk dianalisis 👇")

uploaded_file = st.file_uploader("Unggah Gambar", type=["jpg", "jpeg", "png"])

# ==========================
# Fungsi Prediksi Klasifikasi
# ==========================
def predict_class(img, dataset):
    img_resized = img.resize((224, 224))
    img_array = image.img_to_array(img_resized)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0

    prediction = classifier.predict(img_array)
    class_index = np.argmax(prediction)

    if dataset == "Elektronik":
        labels = ["Laptop", "Mobile", "Supercar"]
    else:
        labels = ["Organic Waste", "Recyclable Waste"]

    return labels[class_index], float(np.max(prediction))

# ==========================
# Tampilan Hasil
# ==========================
if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption="📸 Gambar yang Diunggah", use_container_width=True)

    if mode == "Deteksi Objek (YOLO)":
        st.subheader("🔍 Hasil Deteksi Objek (YOLO)")
        results = yolo_model(img)
        result_img = results[0].plot()  # hasil deteksi dengan bounding box
        st.image(result_img, caption="Hasil Deteksi", use_container_width=True)

    elif mode == "Klasifikasi Gambar":
        st.subheader("🧩 Hasil Klasifikasi Gambar")
        label, prob = predict_class(img, dataset_option)
        st.success(f"**Prediksi:** {label}")
        st.write(f"**Probabilitas:** {prob:.2%}")

else:
    st.info("📤 Silakan unggah gambar terlebih dahulu untuk memulai analisis.")

