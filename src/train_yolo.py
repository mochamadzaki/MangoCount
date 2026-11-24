from pathlib import Path
from ultralytics import YOLO
import shutil


# ==========================================
# KONFIGURASI
# ==========================================

PROJECT_DIR = Path(__file__).resolve().parents[1]

DATA_YAML = Path(
    "/run/media/mchmdzaki/4A688C98688C8483/"
    "UTS_Pengolahan_Citra_Data/dataset/"
    "processed/mango_combined_final/data.yaml"
)

MODEL_NAME = PROJECT_DIR / "yolo11n.pt"

OUTPUT_DIR = PROJECT_DIR / "runs" / "mango_training"

MODEL_OUTPUT = PROJECT_DIR / "models" / "best_mango.pt"


# ==========================================
# FUNGSI UTAMA
# ==========================================

def main():

    print("=" * 55)
    print("TRAINING YOLO11N - DETEKSI BUAH MANGGA")
    print("=" * 55)

    # Pemeriksaan file
    if not DATA_YAML.exists():
        raise FileNotFoundError(
            f"Dataset tidak ditemukan: {DATA_YAML}"
        )

    if not MODEL_NAME.exists():
        raise FileNotFoundError(
            f"Model tidak ditemukan: {MODEL_NAME}"
        )

    # Memuat model
    print("\nMemuat model YOLO11n...")

    model = YOLO(str(MODEL_NAME))

    print("Model berhasil dimuat.")

    # Konfigurasi training
    print("\nKonfigurasi training:")
    print("Epoch       : 30")
    print("Image size  : 512")
    print("Batch       : 2")
    print("Device      : CPU")
    print("Workers     : 0")
    print("Dataset     : 100%")

    # Mulai training
    print("\nMemulai training...\n")

    model.train(

        data=str(DATA_YAML),

        epochs=30,

        imgsz=512,

        batch=2,

        device="cpu",

        workers=0,

        fraction=1.0,

        patience=10,

        cache=False,

        save=True,

        save_period=10,

        plots=True,

        project=str(OUTPUT_DIR),

        name="mango_yolo11n_full",

        exist_ok=False,

        verbose=True,

        seed=42

    )

    # Mengambil lokasi hasil training
    save_dir = Path(model.trainer.save_dir)

    best_model = save_dir / "weights" / "best.pt"

    print("\n" + "=" * 55)
    print("TRAINING SELESAI")
    print("=" * 55)

    # Memeriksa best.pt
    if best_model.exists():

        MODEL_OUTPUT.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        shutil.copy2(
            best_model,
            MODEL_OUTPUT
        )

        print("\nModel terbaik berhasil disimpan.")

        print("Lokasi hasil training:")
        print(best_model)

        print("\nSalinan model:")
        print(MODEL_OUTPUT)

    else:

        print("\nFile best.pt belum ditemukan.")
        print("Periksa folder hasil training.")

    print("\nSemua proses selesai.")


if __name__ == "__main__":
    main()