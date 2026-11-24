from pathlib import Path
from ultralytics import YOLO
import csv
import math


# ==================================================
# 1. KONFIGURASI
# ==================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "best_mango.pt"
)

DATASET_DIR = Path(
    "/run/media/mchmdzaki/4A688C98688C8483/"
    "UTS_Pengolahan_Citra_Data/dataset/"
    "processed/mango_combined_final"
)

IMAGE_DIR = DATASET_DIR / "test" / "images"
LABEL_DIR = DATASET_DIR / "test" / "labels"

OUTPUT_DIR = (
    PROJECT_ROOT
    / "runs"
    / "counting_evaluation"
    / "full_test"
)

IMAGE_SIZE = 512
CONFIDENCE = 0.25
BATCH_SIZE = 2

EXPECTED_IMAGES = {
    "mangoyolo": 300,
    "deepfruits": 18
}


# ==================================================
# 2. MENGHITUNG GROUND TRUTH
# ==================================================

def count_ground_truth(label_path):

    if not label_path.exists():
        raise FileNotFoundError(
            f"Label tidak ditemukan: {label_path}"
        )

    lines = label_path.read_text().splitlines()

    return sum(
        1 for line in lines
        if line.strip()
    )


# ==================================================
# 3. MEMILIH SUMBER DATASET
# ==================================================

def identify_source(image_path):

    filename = image_path.name.lower()

    if filename.startswith("mangoyolo_"):
        return "mangoyolo"

    if filename.startswith("deepfruits_"):
        return "deepfruits"

    raise ValueError(
        f"Sumber gambar tidak dikenali: {filename}"
    )


# ==================================================
# 4. MENGHITUNG METRIK COUNTING
# ==================================================

def calculate_metrics(records):

    if not records:
        return None

    absolute_errors = [
        abs(item["error"])
        for item in records
    ]

    squared_errors = [
        item["error"] ** 2
        for item in records
    ]

    signed_errors = [
        item["error"]
        for item in records
    ]

    mae = sum(absolute_errors) / len(records)

    rmse = math.sqrt(
        sum(squared_errors) / len(records)
    )

    bias = sum(signed_errors) / len(records)

    exact_matches = sum(
        item["error"] == 0
        for item in records
    )

    exact_accuracy = (
        exact_matches / len(records)
    ) * 100

    return {
        "images": len(records),
        "ground_truth": sum(
            item["ground_truth"] for item in records
        ),
        "prediction": sum(
            item["prediction"] for item in records
        ),
        "mae": mae,
        "rmse": rmse,
        "bias": bias,
        "exact_matches": exact_matches,
        "exact_accuracy": exact_accuracy
    }


# ==================================================
# 5. MENYIMPAN HASIL CSV
# ==================================================

def save_csv(records):

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    csv_path = OUTPUT_DIR / "counting_results.csv"

    fieldnames = [
        "source",
        "image",
        "ground_truth",
        "prediction",
        "error",
        "absolute_error"
    ]

    with open(
        csv_path,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames
        )

        writer.writeheader()

        for item in records:

            writer.writerow({
                "source": item["source"],
                "image": item["image"],
                "ground_truth": item["ground_truth"],
                "prediction": item["prediction"],
                "error": item["error"],
                "absolute_error": abs(item["error"])
            })

    print("\nFile CSV berhasil disimpan:")
    print(csv_path)


# ==================================================
# 6. MENYIMPAN RINGKASAN
# ==================================================

def save_summary(results):

    summary_path = OUTPUT_DIR / "counting_summary.txt"

    lines = [
        "EVALUASI COUNTING BUAH MANGGA",
        "=" * 55,
        "",
        "MODEL: best_mango.pt",
        f"Image size: {IMAGE_SIZE}",
        f"Confidence: {CONFIDENCE}",
        f"Device: CPU",
        "",
        "HASIL PER SUMBER DATASET",
        "-" * 55
    ]

    for source, metrics in results.items():

        lines.extend([
            "",
            source.upper(),
            f"Jumlah gambar: {metrics['images']}",
            f"Ground Truth: {metrics['ground_truth']}",
            f"Prediction: {metrics['prediction']}",
            f"MAE: {metrics['mae']:.4f}",
            f"RMSE: {metrics['rmse']:.4f}",
            f"Bias: {metrics['bias']:.4f}",
            f"Exact matches: {metrics['exact_matches']}",
            f"Exact count accuracy: {metrics['exact_accuracy']:.2f}%",
        ])

    summary_path.write_text(
        "\n".join(lines),
        encoding="utf-8"
    )

    print("\nRingkasan berhasil disimpan:")
    print(summary_path)


# ==================================================
# 7. FUNGSI UTAMA
# ==================================================

def main():

    print("=" * 65)
    print("EVALUASI COUNTING SELURUH DATA TESTING")
    print("=" * 65)

    # Pemeriksaan model
    if not MODEL_PATH.exists():
        raise FileNotFoundError(MODEL_PATH)

    if not IMAGE_DIR.exists():
        raise FileNotFoundError(IMAGE_DIR)

    if not LABEL_DIR.exists():
        raise FileNotFoundError(LABEL_DIR)

    # Mengambil seluruh gambar
    image_paths = sorted(
        image
        for image in IMAGE_DIR.iterdir()
        if image.is_file()
        and image.suffix.lower() in {
            ".jpg", ".jpeg", ".png"
        }
    )

    if len(image_paths) != sum(EXPECTED_IMAGES.values()):
        raise ValueError(
            f"Jumlah gambar tidak sesuai: {len(image_paths)}"
        )

    source_counts = {
        source: 0
        for source in EXPECTED_IMAGES
    }

    for image in image_paths:
        source_counts[identify_source(image)] += 1

    for source, expected in EXPECTED_IMAGES.items():

        if source_counts[source] != expected:
            raise ValueError(
                f"Jumlah gambar {source} tidak sesuai: "
                f"{source_counts[source]}"
            )

    print("\nJumlah gambar:")
    print("MangoYOLO :", source_counts["mangoyolo"])
    print("DeepFruits:", source_counts["deepfruits"])
    print("Total     :", len(image_paths))

    # Memuat model
    print("\nMemuat model utama...")

    model = YOLO(str(MODEL_PATH))

    print("Model berhasil dimuat.")

    # Menyiapkan penyimpanan hasil
    all_records = []

    print("\nMemulai proses counting...\n")

    # Proses gambar dalam batch kecil
    for start in range(0, len(image_paths), BATCH_SIZE):

        batch_paths = image_paths[
            start:start + BATCH_SIZE
        ]

        predictions = model.predict(
            source=[str(path) for path in batch_paths],
            imgsz=IMAGE_SIZE,
            conf=CONFIDENCE,
            device="cpu",
            batch=BATCH_SIZE,
            workers=0,
            save=False,
            verbose=False
        )

        for image_path, result in zip(
            batch_paths,
            predictions
        ):

            label_path = (
                LABEL_DIR / f"{image_path.stem}.txt"
            )

            ground_truth = count_ground_truth(
                label_path
            )

            prediction = (
                len(result.boxes)
                if result.boxes is not None
                else 0
            )

            source = identify_source(image_path)

            record = {
                "source": source,
                "image": image_path.name,
                "ground_truth": ground_truth,
                "prediction": prediction,
                "error": prediction - ground_truth
            }

            all_records.append(record)

        completed = min(
            start + BATCH_SIZE,
            len(image_paths)
        )

        print(
            f"Progress: {completed}/{len(image_paths)} gambar",
            flush=True
        )

    # Memisahkan hasil berdasarkan sumber
    grouped_records = {
        source: [
            item for item in all_records
            if item["source"] == source
        ]
        for source in EXPECTED_IMAGES
    }

    # Menghitung metrik
    results = {}

    for source, records in grouped_records.items():

        results[source] = calculate_metrics(records)

    results["overall"] = calculate_metrics(all_records)

    # Menampilkan hasil
    print("\n" + "=" * 85)
    print("HASIL EVALUASI COUNTING")
    print("=" * 85)

    print(
        f"{'DATASET':<16}"
        f"{'IMAGES':<10}"
        f"{'GT':<10}"
        f"{'PRED':<10}"
        f"{'MAE':<12}"
        f"{'RMSE':<12}"
        f"{'BIAS':<12}"
    )

    print("-" * 85)

    for source, metrics in results.items():

        print(
            f"{source:<16}"
            f"{metrics['images']:<10}"
            f"{metrics['ground_truth']:<10}"
            f"{metrics['prediction']:<10}"
            f"{metrics['mae']:<12.4f}"
            f"{metrics['rmse']:<12.4f}"
            f"{metrics['bias']:<12.4f}"
        )

    print("\nExact count accuracy:")

    for source, metrics in results.items():

        print(
            f"{source}: "
            f"{metrics['exact_accuracy']:.2f}% "
            f"({metrics['exact_matches']}/"
            f"{metrics['images']} gambar)"
        )

    # Simpan laporan
    save_csv(all_records)
    save_summary(results)

    print("\n" + "=" * 65)
    print("EVALUASI COUNTING SELESAI")
    print("=" * 65)

    print("Lokasi hasil:")
    print(OUTPUT_DIR)


# ==================================================
# EKSEKUSI
# ==================================================

if __name__ == "__main__":
    main()