# 🦅 Deteksi Burung - Bird Detection System

Sistem deteksi burung menggunakan YOLOv11 (You Only Look Once) untuk mendeteksi dan mengidentifikasi objek burung dalam gambar secara real-time dengan akurasi tinggi.

## 📋 Daftar Isi

- [Fitur Utama](#fitur-utama)
- [Prasyarat](#prasyarat)
- [Instalasi](#instalasi)
- [Penggunaan](#penggunaan)
- [Struktur Proyek](#struktur-proyek)
- [Konfigurasi](#konfigurasi)
- [Training Model](#training-model)
- [Troubleshooting](#troubleshooting)

---

## ✨ Fitur Utama

- **Deteksi Real-Time**: Mendeteksi burung dalam gambar dengan akurasi tinggi menggunakan YOLOv11
- **Antarmuka Web**: Aplikasi Streamlit yang user-friendly dan responsif
- **Pengaturan Fleksibel**: 
  - Slider untuk mengatur sensitivitas deteksi (Confidence Threshold)
  - Slider untuk mengatur eliminasi kotak bertumpuk (IoU Threshold)
- **Gambar Contoh**: Bisa menguji dengan sample images dari folder `testing`
- **Download Hasil**: Unduh gambar hasil deteksi langsung dari aplikasi
- **Dukungan GPU**: Otomatis menggunakan GPU jika tersedia (CUDA), fallback ke CPU
- **Rincian Akurasi**: Lihat confidence score untuk setiap burung yang terdeteksi

---

## 🔧 Prasyarat

Sebelum menginstal, pastikan sistem Anda memenuhi:

- **Python**: 3.8 atau lebih tinggi
- **pip**: Package manager Python
- **Memory**: Minimal 2GB RAM (4GB+ direkomendasikan)
- **GPU** (Opsional): NVIDIA GPU dengan CUDA untuk performa lebih baik

**Cek versi Python:**
```bash
python --version
```

---

## 📦 Instalasi

### 1. Clone Repository

```bash
git clone https://github.com/imamarin/burung-detection.git
cd burung-detection
```

### 2. Buat Virtual Environment (Direkomendasikan)

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Instal Dependencies

```bash
pip install -r requirements.txt
```

**Dependencies yang akan diinstal:**
- **ultralytics** - Framework YOLO
- **opencv-python** - Pemrosesan gambar
- **streamlit** - Framework aplikasi web
- **Pillow** - Library manipulasi gambar
- **torch** - Deep learning framework (instalasi otomatis via ultralytics)

### 4. Verifikasi Instalasi

```bash
python -c "import streamlit; import ultralytics; print('✅ Instalasi berhasil!')"
```

---

## 🚀 Penggunaan

### Menjalankan Aplikasi Web

```bash
streamlit run app.py
```

Aplikasi akan terbuka di browser Anda (default: `http://localhost:8501`)

**Cara Menggunakan:**

1. **Upload Gambar**: 
   - Klik tombol "Pilih gambar burung" untuk upload gambar dari komputer
   - Atau gunakan checkbox "Coba gambar contoh" untuk testing dengan sample images

2. **Pengaturan Deteksi** (Sidebar kiri):
   - **Sensitivitas Deteksi**: Turunkan jika burung belum terdeteksi, naikkan jika ada false positives
   - **Eliminasi Kotak Bertumpuk (IoU)**: Mengontrol agresivitas penghapusan duplikat deteksi

3. **Lihat Hasil**:
   - Gambar asli dan hasil deteksi ditampilkan berdampingan
   - Jumlah burung yang terdeteksi ditampilkan di atas gambar

4. **Download Hasil**:
   - Klik tombol "📥 Download Hasil Deteksi" untuk menyimpan gambar hasil

5. **Rincian Akurasi**:
   - Klik "📊 Lihat rincian akurasi" untuk melihat confidence score setiap deteksi

---

## 📂 Struktur Proyek

```
burung-detection/
├── app.py                 # Aplikasi Streamlit utama
├── train.py              # Script untuk training model
├── main.py               # File alternatif (copy dari app.py)
├── requirements.txt      # Daftar dependencies
├── data.yaml             # Konfigurasi dataset untuk training
├── classes.txt           # Daftar kelas objek (hanya "burung")
├── yolo11n.pt            # Model YOLO11 nano (pre-trained)
├── notes.json            # Catatan dan konfigurasi tambahan
├── README.md             # File dokumentasi ini
├── training/             # Folder dataset untuk training
│   └── images/           # Gambar training
├── testing/              # Folder dataset untuk testing
│   └── images/           # Gambar testing/contoh
├── runs/                 # Folder hasil training
│   ├── detect/
│   │   └── train_burung/ # Folder hasil training
│   │       └── weights/
│   │           └── best.pt  # Model terbaik hasil training
│   └── ...
└── __pycache__/          # Cache Python (auto-generated)
```

---

## ⚙️ Konfigurasi

### File `data.yaml`

Konfigurasi dataset untuk training:

```yaml
path: .                  # Root direktori dataset
train: training/images   # Path gambar training
val: testing/images      # Path gambar validasi
test: testing/images     # Path gambar testing

nc: 1                    # Jumlah kelas (1 untuk burung)
names:
  0: burung              # Nama kelas
```

### File `app.py` - Parameter Penting

Anda dapat mengubah parameter berikut di sidebar aplikasi:

| Parameter | Default | Range | Fungsi |
|-----------|---------|-------|--------|
| `conf_threshold` | 0.10 | 0.01 - 1.00 | Confidence minimum untuk deteksi |
| `iou_threshold` | 0.45 | 0.10 - 0.95 | IoU threshold untuk NMS |
| `imgsz` | 640 | - | Ukuran resolusi gambar input |

---

## 🎓 Training Model

Jika ingin melatih ulang model dengan dataset Anda sendiri:

### Persiapan Dataset

1. **Buat folder struktur:**
   ```
   training/images/     # Gambar untuk training
   testing/images/      # Gambar untuk validasi
   ```

2. **Format dataset**: 
   - Gunakan format YOLO (text annotations)
   - Setiap gambar harus memiliki file `.txt` dengan koordinat bounding box

### Menjalankan Training

```bash
python train.py
```

**Parameter training di `train.py`:**

| Parameter | Nilai | Keterangan |
|-----------|-------|-----------|
| `epochs` | 50 | Jumlah epoch training |
| `imgsz` | 640 | Resolusi gambar (standard YOLO) |
| `batch` | 2 | Ukuran batch (sesuaikan dengan VRAM GPU) |
| `device` | 0 atau 'cpu' | 0 = GPU, 'cpu' = CPU |
| `workers` | 0 | Workers untuk data loading (0 untuk Windows) |

**Hasil training disimpan di:**
```
runs/detect/train_burung/
├── weights/
│   ├── best.pt          # Model terbaik
│   └── last.pt          # Model terakhir
├── results.png          # Grafik hasil training
└── ...
```

### Menggunakan Model Custom

Setelah training, aplikasi akan otomatis menggunakan `best.pt` jika ditemukan di `runs/detect/train_burung/weights/`.

---

## 🐛 Troubleshooting

### Error: "Model tidak ditemukan"

**Solusi:**
- Pastikan file `yolo11n.pt` ada di root folder
- Atau jalankan `train.py` terlebih dahulu untuk generate `best.pt`
- Download manual: `https://github.com/ultralytics/assets/releases/download/v8.2.0/yolo11n.pt`

```bash
# Download model
wget https://github.com/ultralytics/assets/releases/download/v8.2.0/yolo11n.pt
```

### Deteksi tidak akurat

**Solusi:**
- Turunkan slider "Sensitivitas Deteksi" di sidebar
- Coba nilai conf_threshold = 0.05 - 0.15
- Jika model masih tidak akurat, training ulang dengan dataset yang lebih baik

### Error: "ModuleNotFoundError"

**Solusi:**
```bash
# Pastikan virtual environment aktif
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# Reinstall dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### Aplikasi berjalan lambat

**Solusi:**
- Gunakan GPU jika tersedia (cek dengan `torch.cuda.is_available()`)
- Kurangi ukuran batch di `train.py`
- Optimasi gambar input (resize ke 640x640)

### Streamlit tidak merespons

**Solusi:**
```bash
# Clear cache Streamlit
streamlit cache clear

# Jalankan ulang
streamlit run app.py
```

### CUDA tidak terdeteksi

**Solusi:**
```bash
# Cek pytorch installation
pip show torch torchvision torchaudio

# Reinstall pytorch dengan CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

---

## 📊 Contoh Output

```
✅ Terdeteksi 3 burung!

[Gambar Asli] | [Hasil Deteksi]

📊 Rincian Akurasi:
| No | Objek | Akurasi |
|----|-------|---------|
| 1  | Burung| 94.2%   |
| 2  | Burung| 87.5%   |
| 3  | Burung| 91.3%   |

[📥 Download Hasil Deteksi]
```

---

## 📝 Catatan Penting

- Model YOLOv11n adalah model nano (ukuran kecil, cepat tapi akurasi sedang)
- Untuk akurasi lebih tinggi, gunakan model lebih besar (yolo11s, yolo11m, yolo11l)
- Dataset training yang besar dan berkualitas akan meningkatkan akurasi model
- Backup model terbaik Anda sebelum melatih ulang

---

## 🔗 Referensi

- [YOLOv11 Documentation](https://docs.ultralytics.com/models/yolov11/)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
- [Ultralytics GitHub](https://github.com/ultralytics/ultralytics)

---

## 📄 Lisensi

Proyek ini menggunakan model YOLO dari [Ultralytics](https://github.com/ultralytics/ultralytics) yang dilisensikan di bawah AGPL-3.0 License.

---

## 👨‍💻 Penulis

**@imamarin**

---

## 🤝 Kontribusi

Kontribusi welcome! Silakan buat issue atau pull request untuk improvement.

---

**Terakhir diupdate**: September 2026
