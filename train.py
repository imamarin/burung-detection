import torch
from ultralytics import YOLO
import os

model = YOLO('yolo11n.pt')

# Pilih GPU jika CUDA tersedia, jika tidak fallback ke CPU
device = 0 if torch.cuda.is_available() else 'cpu'
print(f"[INFO] Menggunakan device: {device} ({'GPU' if device == 0 else 'CPU'})")

print("[INFO] Memulai training...")
results = model.train(
    data="data.yaml",
    epochs=50,          # Jumlah epoch / iterasi training
    imgsz=640,          # Resolusi gambar (standard YOLO 640x640)
    batch=2,            # Ukuran batch (disesuaikan dengan jumlah data & VRAM)
    device=device,      # 0 untuk GPU Nvidia, atau 'cpu'
    workers=0,          # 0 direkomendasikan di Windows untuk dataset kecil
    name="train_burung",   # Nama folder hasil training (disimpan di runs/detect/train_burung)
    exist_ok=True       # Timpa folder yang ada jika sudah pernah dijalankan
)

