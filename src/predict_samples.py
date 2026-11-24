from pathlib import Path
from ultralytics import YOLO
import random


# ==========================================
# KONFIGURASI
# ==========================================

PROJECT_DIR = Path(__file__).resolve().parents[1]

MODEL_PATH = PROJECT_DIR / "models" / "best_mango.pt"

DATA_ROOT = Path(
    "/run/media/mchmdzaki/4A688C98688C8483/"
    "UTS_Pengolahan_Citra_Data/dataset"
)

OUTPUT_DIR = PROJECT_DIR / "runs" / "mango_predictions"


# ==========================================
# SUMBER GAMBAR
# ==========================================

DATASETS = {
    "MangoYOLO": (
        DATA_ROOT
        / "processed/mango_combined_final/test/images"
    ),

    "DeepFruits": (
        DATA_ROOT
        / "processed/mango_combined_final/test/images"
    )
}


# ==========================================
# PENGUJIAN DETEKSI
# ==========================================

def main():

    print("=" * 50)
    print("PENGUJIAN DETEKSI BUAH MANGGA")
    print("=" * 50)

    if not MODEL_PATH.exists():
        raise FileNotFoundError(MODEL_PATH)

    print("\nMemuat model...")
    model = YOLO(str(MODEL_PATH))

    print("Model berhasil dimuat.")

    random.seed(42)

    selected_images = []

    test_dir = next(iter(DATASETS.values()))

    for source in DATASETS:

        source_images = [
            path
            for path in test_dir.glob(f"{source.lower()}_*")
            if path.suffix.lower() in {
                ".jpg", ".jpeg", ".png"
            }
        ]

        if not source_images:
            print(f"\nTidak menemukan gambar: {source}")
            continue

        samples = random.sample(
            source_images,
            min(3, len(source_images))
        )

        print(f"\n{source}: {len(samples)} gambar dipilih.")

        selected_images.extend(samples)

    if not selected_images:
        raise RuntimeError("Tidak ada gambar untuk diuji.")

    print("\nMulai deteksi...\n")

    results = model.predict(
        source=[str(image) for image in selected_images],
        imgsz=512,
        conf=0.25,
        device="cpu",
        save=True,
        project=str(OUTPUT_DIR),
        name="test_samples",
        exist_ok=True,
        verbose=False
    )

    print("\n===== HASIL DETEKSI =====")

    for result in results:

        count = len(result.boxes)

        print("Gambar:", Path(result.path).name)
        print("Mangga terdeteksi:", count)
        print("-" * 35)

    print("\nPengujian selesai.")
    print("Hasil tersimpan di:")
    print(OUTPUT_DIR / "test_samples")


if __name__ == "__main__":
    main()