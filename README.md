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
│   ├── images/           # Gambar training
│   └── labels/           # Label YOLO format (.txt)
├── testing/              # Folder dataset untuk testing
│   ├── images/           # Gambar testing/contoh
│   └── labels/           # Label YOLO format (.txt)
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

### Persiapan Dataset menggunakan Label Studio

Label Studio adalah platform anotasi data berbasis web yang memudahkan labeling gambar untuk object detection.

#### **Langkah 1: Install dan Setup Label Studio**

```bash
pip install label-studio
```

Jalankan Label Studio:
```bash
label-studio
```

Label Studio akan terbuka di `http://localhost:8080` di browser Anda.

#### **Langkah 2: Buat Project dan Import Gambar**

1. **Login** ke Label Studio (buat akun jika belum punya)
2. **Create Project**:
   - Klik "Create" → "Labeling Project"
   - Beri nama: "Burung Detection" atau nama lainnya
   - Pilih tipe task: **Object Detection with Bounding Box**

3. **Import Gambar**:
   - Klik "Import" → Upload atau drag-drop gambar burung
   - Gambar harus format JPG, PNG, atau format image umum lainnya
   - Anda bisa upload dari folder yang sama kemudian split untuk training/testing

#### **Langkah 3: Anotasi Gambar (Bounding Box)**

Untuk setiap gambar:
1. Klik gambar untuk mulai anotasi
2. **Buat Bounding Box**:
   - Click dan drag untuk membuat kotak di sekitar burung
   - Beri label: `burung` (harus sama dengan nama kelas di `classes.txt`)
   - Repeat untuk semua burung dalam gambar
3. Klik **Submit** atau **Save** setelah selesai

**Tips:**
- Usahakan bounding box yang pas menutup burung tanpa terlalu banyak space kosong
- Untuk burung yang overlapping, pisahkan dengan bounding box terpisah
- Anotasi minimal 50-100 gambar untuk hasil yang baik

#### **Langkah 4: Export Dataset dalam Format YOLO**

Setelah selesai anotasi:

1. **Export Project**:
   - Klik Menu (⋮) → **Export**
   - Pilih format: **YOLO**
   - Klik **Export** untuk download

2. **File yang Dihasilkan**:
   ```
   exported_project/
   ├── images/          # Folder gambar
   │   ├── image1.jpg
   │   ├── image2.jpg
   │   └── ...
   └── labels/          # Folder label YOLO
       ├── image1.txt
       ├── image2.txt
       └── ...
   ```

   Struktur file `.txt` (Label Studio format YOLO):
   ```
   <class_id> <x_center> <y_center> <width> <height>
   ```
   
   Contoh untuk 1 burung dalam gambar 640x480:
   ```
   0 0.5 0.4 0.3 0.25
   ```
   - `0` = class id untuk "burung"
   - Koordinat dalam range 0-1 (normalized)

#### **Langkah 5: Split Dataset ke Training dan Testing**

Ekstrak file dari Label Studio, lalu split:

```bash
# Struktur setelah ekstrak:
exported_project/
├── images/
└── labels/

# Buat folder struktur proyek
mkdir -p training/images training/labels
mkdir -p testing/images testing/labels
```

Split manual atau menggunakan script Python:

```python
import os
import shutil
from sklearn.model_selection import train_test_split

# Path ke folder hasil export Label Studio
source_images = "exported_project/images"
source_labels = "exported_project/labels"

# Path tujuan
train_images = "training/images"
train_labels = "training/labels"
test_images = "testing/images"
test_labels = "testing/labels"

# Buat folder jika belum ada
os.makedirs(train_images, exist_ok=True)
os.makedirs(train_labels, exist_ok=True)
os.makedirs(test_images, exist_ok=True)
os.makedirs(test_labels, exist_ok=True)

# Dapatkan list gambar
images = [f for f in os.listdir(source_images) if f.endswith(('.jpg', '.jpeg', '.png'))]

# Split 80% training, 20% testing
train_imgs, test_imgs = train_test_split(images, test_size=0.2, random_state=42)

# Copy gambar dan label ke folder training
for img in train_imgs:
    label_name = os.path.splitext(img)[0] + '.txt'
    
    shutil.copy(os.path.join(source_images, img), os.path.join(train_images, img))
    if os.path.exists(os.path.join(source_labels, label_name)):
        shutil.copy(os.path.join(source_labels, label_name), os.path.join(train_labels, label_name))

# Copy gambar dan label ke folder testing
for img in test_imgs:
    label_name = os.path.splitext(img)[0] + '.txt'
    
    shutil.copy(os.path.join(source_images, img), os.path.join(test_images, img))
    if os.path.exists(os.path.join(source_labels, label_name)):
        shutil.copy(os.path.join(source_labels, label_name), os.path.join(test_labels, label_name))

print(f"✅ Split selesai!")
print(f"Training: {len(train_imgs)} gambar")
print(f"Testing: {len(test_imgs)} gambar")
```

Jalankan script:
```bash
python split_dataset.py
```

#### **Langkah 6: Verifikasi Struktur Dataset**

```
burung-detection/
├── training/
│   ├── images/
│   │   ├── bird1.jpg
│   │   ├── bird2.jpg
│   │   └── ...
│   └── labels/
│       ├── bird1.txt
│       ├── bird2.txt
│       └── ...
├── testing/
│   ├── images/
│   │   ├── bird_test1.jpg
│   │   ├── bird_test2.jpg
│   │   └── ...
│   └── labels/
│       ├── bird_test1.txt
│       ├── bird_test2.txt
│       └── ...
└── data.yaml
```

**Verifikasi isi file label:**
```bash
# Lihat isi file label (example)
cat training/labels/bird1.txt

# Output contoh:
# 0 0.512 0.456 0.298 0.234
```

---

### Menjalankan Training

Setelah dataset siap, jalankan training:

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
- Pastikan dataset memiliki variasi burung (berbagai pose, ukuran, latar belakang)

### Error saat menggunakan Label Studio

**Solusi:**
```bash
# Pastikan Label Studio sudah diinstal
pip install --upgrade label-studio

# Jika port 8080 sudah terpakai, gunakan port lain
label-studio --port 8081
```

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

### File label tidak terbaca

**Solusi:**
- Pastikan file `.txt` berada di folder `labels/` yang sejajar dengan folder `images/`
- Setiap gambar harus memiliki file label dengan nama yang sama (hanya extension berbeda)
- Contoh: `bird1.jpg` → `bird1.txt`
- Jika tidak ada anotasi untuk gambar tertentu, buat file `.txt` kosong untuk gambar tersebut

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
- Label Studio export format YOLO sudah support, tinggal copy ke folder `training/images` dan `testing/images`
- Minimal 50 gambar untuk training yang decent, 100+ untuk hasil lebih baik

---

## 🔗 Referensi

- [Label Studio Documentation](https://labelstud.io/guide/index.html)
- [Label Studio Object Detection](https://labelstud.io/guide/tasks.html#Object%20detection)
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
