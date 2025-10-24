import streamlit as st
from ultralytics import YOLO
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import io

# ==========================
# Konfigurasi Halaman
# ==========================
st.set_page_config(
    page_title="Dashboard Deteksi & Klasifikasi Citra",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================
# Fungsi Load Models
# ==========================
@st.cache_resource
def load_models():
    yolo_model = YOLO("model/Yumnaa Alifah_Laporan 4.pt")  # model deteksi objek
    classifier = tf.keras.models.load_model("model/classifier_model.h5")  # model klasifikasi
    return yolo_model, classifier

yolo_model, classifier = load_models()

# ==========================
# UI Header
# ==========================
st.title("🧠 Dashboard Deteksi & Klasifikasi Citra")
st.markdown("""
Aplikasi ini menggunakan dua model deep learning:
1. **YOLOv8 (.pt)** untuk *deteksi objek (laptop, mobile, supercar)*  
2. **CNN (.h5)** untuk *klasifikasi jenis sampah (plastik, kertas, logam, lainnya)*
""")

# ==========================
# Sidebar Menu
# ==========================
st.sidebar.header("⚙️ Pengaturan Mode")
mode = st.sidebar.radio(
    "Pilih Mode:",
    ["Deteksi Objek (YOLO)", "Klasifikasi Sampah (CNN)"],
    index=0
)

st.sidebar.markdown("---")
st.sidebar.write("📤 Unggah gambar Anda di bawah:")

uploaded_file = st.sidebar.file_uploader("Pilih gambar...", type=["jpg", "jpeg", "png"])

# ==========================
# Main Content
# ==========================
if uploaded_file is not None:
    # Baca dan tampilkan gambar
    image_data = uploaded_file.read()
    img = Image.open(io.BytesIO(image_data)).convert("RGB")

    st.image(img, caption="🖼️ Gambar yang Diupload", use_container_width=True)
    st.markdown("---")

    if mode == "Deteksi Objek (YOLO)":
        st.subheader("🔍 Hasil Deteksi Objek")

        with st.spinner("Model YOLO sedang memproses gambar..."):
            results = yolo_model(img)
            result_img = results[0].plot()  # hasil gambar deteksi
            boxes = results[0].boxes

        st.image(result_img, caption="📦 Hasil Deteksi", use_container_width=True)

        if boxes is not None and len(boxes) > 0:
            st.success("✅ Objek berhasil terdeteksi!")
            data = []
            for box in boxes:
                cls_id = int(box.cls[0])
                label = yolo_model.names[cls_id]
                conf = float(box.conf[0])
                data.append({"Label": label, "Kepercayaan": f"{conf:.2f}"})
            st.table(data)
        else:
            st.warning("⚠️ Tidak ada objek yang terdeteksi pada gambar ini.")

    elif mode == "Klasifikasi Sampah (CNN)":
        st.subheader("🧾 Hasil Klasifikasi Sampah")

        class_labels = ["Sampah Plastik", "Sampah Kertas", "Sampah Logam", "Lainnya"]

        with st.spinner("Model CNN sedang memproses gambar..."):
            img_resized = img.resize((224, 224))
            img_array = image.img_to_array(img_resized)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = img_array / 255.0

            prediction = classifier.predict(img_array)
            class_index = np.argmax(prediction)
            confidence = np.max(prediction)

        st.success("✅ Klasifikasi berhasil dilakukan!")

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Kategori Sampah", class_labels[class_index])
        with col2:
            st.metric("Tingkat Kepercayaan", f"{confidence * 100:.2f}%")

        st.progress(float(confidence))
        st.markdown(f"📊 **Model memprediksi bahwa gambar ini adalah:** `{class_labels[class_index]}`")

else:
    st.info("👈 Silakan unggah gambar di sidebar untuk memulai analisis.")
