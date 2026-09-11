import torch
from ultralytics import YOLO
import os

def main():
    # 1. Cek ketersediaan perangkat (GPU / CPU)
    if torch.cuda.is_available():
        device = 0
        print(f"[INFO] Menggunakan GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = "cpu"
        print("[INFO] GPU tidak terdeteksi / PyTorch CPU version, menggunakan CPU.")

    # 2. Inisialisasi model YOLO pre-trained
    # Anda bisa memilih: 'yolo11n.pt' (terbaru) atau 'yolov8n.pt'
    # 'n' (nano) adalah model teringan dan tercepat
    model_name = "yolo11n.pt"
    print(f"[INFO] Memuat pre-trained model: {model_name}...")
    model = YOLO(model_name)

    # 3. Path ke data.yaml
    data_yaml_path = os.path.abspath("data.yaml")

    # 4. Mulai proses training
    print("[INFO] Memulai training...")
    results = model.train(
        data=data_yaml_path,
        epochs=50,          # Jumlah epoch / iterasi training
        imgsz=640,          # Resolusi gambar (standard YOLO 640x640)
        batch=2,            # Ukuran batch (disesuaikan dengan jumlah data & VRAM)
        device=device,      # 0 untuk GPU Nvidia, atau 'cpu'
        workers=0,          # 0 direkomendasikan di Windows untuk dataset kecil
        name="train_burung",   # Nama folder hasil training (disimpan di runs/detect/train_burung)
        exist_ok=True       # Timpa folder yang ada jika sudah pernah dijalankan
    )

    best_model_path = os.path.join(results.save_dir, "weights", "best.pt")
    print("\n" + "="*50)
    print("[INFO] Training selesai!")
    print(f"[INFO] Bobot model terbaik disimpan di: {best_model_path}")
    print("="*50)

if __name__ == "__main__":
    # Wajib menggunakan __name__ == '__main__' pada Windows untuk multiprocessing
    main()
