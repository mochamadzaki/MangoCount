from pathlib import Path
from ultralytics import YOLO
import shutil


# ==================================================
# KONFIGURASI PROYEK
# ==================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Model pretrained
BASE_MODEL = PROJECT_ROOT / "yolo11n.pt"

# Dataset MangoYOLO saja
DATASET_DIR = Path(
    "/run/media/mchmdzaki/4A688C98688C8483/"
    "UTS_Pengolahan_Citra_Data/dataset/raw/"
    "mango_detection_australia/data/yolo_converted"
)

DATA_YAML = DATASET_DIR / "data.yaml"

# Folder hasil training
OUTPUT_DIR = (
    PROJECT_ROOT
    / "runs"
    / "mango_training"
)

# Lokasi model hasil training
MODEL_OUTPUT = (
    PROJECT_ROOT
    / "models"
    / "best_mangoyolo_only.pt"
)


# ==================================================
# PEMERIKSAAN DATASET
# ==================================================

def check_dataset():

    print("\n===== PEMERIKSAAN DATASET =====")

    if not DATA_YAML.exists():
        raise FileNotFoundError(
            f"data.yaml tidak ditemukan: {DATA_YAML}"
        )

    for split in ["train", "valid", "test"]:

        image_dir = DATASET_DIR / split / "images"
        label_dir = DATASET_DIR / split / "labels"

        if not image_dir.exists() or not label_dir.exists():
            raise FileNotFoundError(
                f"Folder {split} tidak lengkap."
            )

        images = [
            p for p in image_dir.iterdir()
            if p.is_file()
        ]

        labels = list(label_dir.glob("*.txt"))

        if len(images) != len(labels):
            raise ValueError(
                f"Jumlah gambar dan label {split} berbeda."
            )

        print(f"{split.upper():10}: {len(images)} gambar")

    print("\nKonfigurasi dataset ditemukan.")


# ==================================================
# TRAINING YOLO11N
# ==================================================

def main():

    print("=" * 55)
    print("EKSPERIMEN 2 - MANGOYOLO AUSTRALIA")
    print("=" * 55)

    # Periksa model
    if not BASE_MODEL.exists():
        raise FileNotFoundError(
            f"Model dasar tidak ditemukan: {BASE_MODEL}"
        )

    # Periksa dataset
    check_dataset()

    print("\nMemuat YOLO11n pretrained...")

    model = YOLO(str(BASE_MODEL))

    print("Model berhasil dimuat.")

    print("\n===== KONFIGURASI TRAINING =====")
    print("Dataset    : MangoYOLO Australia")
    print("Epoch      : 30")
    print("Image size : 512")
    print("Batch      : 2")
    print("Device     : CPU")
    print("Workers    : 0")

    print("\nMemulai training...\n")

    # Training
    model.train(

        data=str(DATA_YAML),

        epochs=30,

        imgsz=512,

        batch=2,

        device="cpu",

        workers=0,

        patience=10,

        fraction=1.0,

        cache=False,

        save=True,

        save_period=10,

        plots=True,

        project=str(OUTPUT_DIR),

        name="mangoyolo_only_v1",

        exist_ok=False,

        verbose=True,

        seed=42

    )

    # Mengambil lokasi hasil training
    save_dir = Path(model.trainer.save_dir)

    best_model = (
        save_dir
        / "weights"
        / "best.pt"
    )

    print("\n" + "=" * 55)
    print("TRAINING SELESAI")
    print("=" * 55)

    if not best_model.exists():
        raise FileNotFoundError(
            f"best.pt tidak ditemukan: {best_model}"
        )

    # Menyimpan model dengan nama berbeda
    MODEL_OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    shutil.copy2(
        best_model,
        MODEL_OUTPUT
    )

    print("\nModel terbaik berhasil disimpan.")

    print("Hasil training:")
    print(best_model)

    print("\nSalinan model eksperimen 2:")
    print(MODEL_OUTPUT)

    print("\nEksperimen 2 berhasil.")


# ==================================================
# EKSEKUSI
# ==================================================

if __name__ == "__main__":
    main()