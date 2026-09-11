import os
import io
from PIL import Image
import streamlit as st
import torch
from ultralytics import YOLO

# Konfigurasi Halaman (Centered agar tampilan rapi & simpel)
st.set_page_config(
    page_title="Deteksi Burung",
    page_icon="🦅",
    layout="centered"
)

# Judul & Deskripsi Sederhana
st.title("🦅 Deteksi Burung")
st.write("Unggah gambar untuk mendeteksi objek burung secara otomatis.")

# Memuat model secara otomatis
@st.cache_resource
def load_model():
    model_paths = [
        os.path.join("runs", "detect", "train_burung", "weights", "best.pt"),
        os.path.join("runs", "train_burung", "weights", "best.pt"),
        "yolo11n.pt"
    ]
    for path in model_paths:
        if os.path.exists(path):
            return YOLO(path)
    return None

model = load_model()

if model is None:
    st.error("Model tidak ditemukan. Pastikan file `best.pt` atau `yolo11n.pt` tersedia.")
    st.stop()

# Sidebar Minimalis
with st.sidebar:
    st.header("Pengaturan")
    conf_threshold = st.slider(
        "Sensitivitas Deteksi (Confidence)",
        min_value=0.01,
        max_value=1.00,
        value=0.10,
        step=0.01,
        help="Turunkan nilai jika burung belum terdeteksi, naikkan jika ada deteksi yang salah."
    )

    iou_threshold = st.slider(
        "Eliminasi Kotak Bertumpuk (IoU)",
        min_value=0.10,
        max_value=0.95,
        value=0.45,
        step=0.05,
        help="Mengeliminasi kotak yang tumpang tindih (Non-Maximum Suppression). Nilai lebih kecil akan lebih agresif menghapus kotak ganda."
    )
    
    # Opsi gambar contoh
    sample_dir = os.path.join("testing", "images")
    sample_images = []
    if os.path.exists(sample_dir):
        sample_images = [f for f in os.listdir(sample_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    selected_sample = None
    if sample_images:
        st.write("---")
        use_sample = st.checkbox("Coba gambar contoh")
        if use_sample:
            selected_sample = st.selectbox("Pilih gambar contoh:", sample_images)

# Input Gambar
image_input = None
if selected_sample:
    image_input = Image.open(os.path.join(sample_dir, selected_sample)).convert("RGB")
else:
    uploaded_file = st.file_uploader("Pilih gambar burung (JPG/PNG):", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image_input = Image.open(uploaded_file).convert("RGB")

# Tampilan Hasil Deteksi
if image_input is not None:
    # Proses deteksi dengan confidence threshold & IoU threshold
    device = 0 if torch.cuda.is_available() else "cpu"
    results = model.predict(
        source=image_input,
        conf=conf_threshold,
        iou=iou_threshold,
        device=device,
        verbose=False
    )
    res = results[0]
    boxes = res.boxes
    total_burung = len(boxes)

    # Siapkan gambar hasil
    res_bgr = res.plot()
    res_rgb = res_bgr[:, :, ::-1]  # Konversi BGR ke RGB
    result_image = Image.fromarray(res_rgb)

    st.write("")
    # Notifikasi status
    if total_burung > 0:
        st.success(f"✅ Terdeteksi **{total_burung}** burung!")
    else:
        st.warning("⚠️ Belum ada burung terdeteksi. Coba turunkan slider 'Sensitivitas Deteksi' di sebelah kiri.")

    # Tampilkan Gambar Berdampingan
    col1, col2 = st.columns(2)
    with col1:
        st.image(image_input, caption="Gambar Asli", width="stretch")
    with col2:
        st.image(result_image, caption="Hasil Deteksi", width="stretch")

    # Tombol Download Hasil
    buf = io.BytesIO()
    result_image.save(buf, format="JPEG")
    st.download_button(
        label="📥 Download Hasil Deteksi",
        data=buf.getvalue(),
        file_name="hasil_deteksi_burung.jpg",
        mime="image/jpeg",
        width="stretch"
    )

    # Detail Akurasi (Opsional, dalam expander agar tidak penuh)
    if total_burung > 0:
        with st.expander("📊 Lihat rincian akurasi"):
            data = []
            for i, box in enumerate(boxes):
                score = float(box.conf[0]) * 100
                data.append({
                    "No": i + 1,
                    "Objek": "Burung",
                    "Akurasi": f"{score:.1f}%"
                })
            st.table(data)
else:
    st.info("💡 Unggah gambar di atas untuk melihat hasil deteksi.")
