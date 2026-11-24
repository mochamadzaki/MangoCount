from pathlib import Path
from ultralytics import YOLO
import random


# ==========================================
# KONFIGURASI
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = PROJECT_ROOT / "models" / "best_mango.pt"

DATASET_DIR = Path(
    "/run/media/mchmdzaki/4A688C98688C8483/"
    "UTS_Pengolahan_Citra_Data/dataset/"
    "processed/mango_combined_final"
)

IMAGE_DIR = DATASET_DIR / "test" / "images"
LABEL_DIR = DATASET_DIR / "test" / "labels"

OUTPUT_DIR = PROJECT_ROOT / "runs" / "count_comparison"

SAMPLES_PER_SOURCE = 3
CONFIDENCE = 0.25


# ==========================================
# MENGHITUNG GROUND TRUTH
# ==========================================

def count_ground_truth(label_path):

    if not label_path.exists():
        raise FileNotFoundError(label_path)

    return sum(
        1
        for line in label_path.read_text().splitlines()
        if line.strip()
    )


# ==========================================
# MEMILIH GAMBAR
# ==========================================

def select_samples():

    random.seed(42)

    selected = []

    sources = ["mangoyolo", "deepfruits"]

    for source in sources:

        images = [
            image
            for image in IMAGE_DIR.iterdir()
            if image.is_file()
            and image.stem.startswith(source + "_")
            and image.suffix.lower() in {
                ".jpg", ".jpeg", ".png"
            }
        ]

        if not images:
            print(f"Tidak ada gambar untuk sumber: {source}")
            continue

        samples = random.sample(
            images,
            min(SAMPLES_PER_SOURCE, len(images))
        )

        selected.extend(samples)

    return selected


# ==========================================
# PERBANDINGAN GROUND TRUTH DAN PREDICTION
# ==========================================

def main():

    print("=" * 65)
    print("PERBANDINGAN GROUND TRUTH DAN PREDICTION")
    print("=" * 65)

    # Pemeriksaan file
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model tidak ditemukan: {MODEL_PATH}"
        )

    if not IMAGE_DIR.exists():
        raise FileNotFoundError(
            f"Folder gambar tidak ditemukan: {IMAGE_DIR}"
        )

    if not LABEL_DIR.exists():
        raise FileNotFoundError(
            f"Folder label tidak ditemukan: {LABEL_DIR}"
        )

    # Memuat model
    print("\nMemuat model YOLO11n...")

    model = YOLO(str(MODEL_PATH))

    print("Model berhasil dimuat.")

    # Memilih gambar
    samples = select_samples()

    if not samples:
        raise RuntimeError("Tidak ada gambar yang bisa diuji.")

    print("\nJumlah gambar diuji:", len(samples))

    # Prediksi
    results = model.predict(
        source=[str(image) for image in samples],
        imgsz=512,
        conf=CONFIDENCE,
        device="cpu",
        save=True,
        project=str(OUTPUT_DIR),
        name="comparison",
        exist_ok=True,
        verbose=False
    )

    # Statistik
    total_gt = 0
    total_pred = 0
    total_absolute_error = 0

    print("\n" + "=" * 100)

    print(
        f"{'SUMBER':<14}"
        f"{'GROUND TRUTH':<16}"
        f"{'PREDICTION':<14}"
        f"{'SELISIH':<12}"
        f"NAMA GAMBAR"
    )

    print("=" * 100)

    # Membandingkan setiap gambar
    for result in results:

        image_path = Path(result.path)

        label_path = (
            LABEL_DIR / f"{image_path.stem}.txt"
        )

        # Jumlah sebenarnya
        ground_truth = count_ground_truth(label_path)

        # Jumlah hasil deteksi
        prediction = len(result.boxes)

        # Selisih
        difference = prediction - ground_truth

        absolute_error = abs(difference)

        # Menentukan sumber
        if image_path.stem.startswith("deepfruits_"):
            source = "DeepFruits"
        else:
            source = "MangoYOLO"

        # Menampilkan hasil per gambar
        print(
            f"{source:<14}"
            f"{ground_truth:<16}"
            f"{prediction:<14}"
            f"{difference:<12}"
            f"{image_path.name}"
        )

        # Memperbarui statistik
        total_gt += ground_truth
        total_pred += prediction
        total_absolute_error += absolute_error

    # ======================================
    # RINGKASAN
    # ======================================

    mae = total_absolute_error / len(results)

    print("\n" + "=" * 65)
    print("RINGKASAN HASIL")
    print("=" * 65)

    print("Total Ground Truth :", total_gt)
    print("Total Prediction   :", total_pred)

    print(
        "Selisih Total      :",
        total_pred - total_gt
    )

    print(
        "Total Absolute Error:",
        total_absolute_error
    )

    print(
        f"MAE per gambar     : {mae:.4f}"
    )

    print("\nPreview tersimpan di:")
    print(OUTPUT_DIR / "comparison")


# ==========================================
# EKSEKUSI PROGRAM
# ==========================================

if __name__ == "__main__":
    main()