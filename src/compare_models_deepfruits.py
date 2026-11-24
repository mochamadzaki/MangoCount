from pathlib import Path
from ultralytics import YOLO
import shutil


# ==========================================
# KONFIGURASI
# ==========================================

ROOT = Path(__file__).resolve().parents[1]

MODEL_GABUNGAN = ROOT / "models" / "best_mango.pt"
MODEL_MANGOYOLO = ROOT / "models" / "best_mangoyolo_only.pt"

COMBINED_DIR = Path(
    "/run/media/mchmdzaki/4A688C98688C8483/"
    "UTS_Pengolahan_Citra_Data/dataset/"
    "processed/mango_combined_final"
)

OUTPUT_ROOT = ROOT / "outputs" / "deepfruits_eval_dataset"

SOURCE_IMAGES = COMBINED_DIR / "test" / "images"
SOURCE_LABELS = COMBINED_DIR / "test" / "labels"

DATA_YAML = OUTPUT_ROOT / "data.yaml"
RESULTS_DIR = ROOT / "runs" / "deepfruits_model_comparison"


# ==========================================
# MENYIAPKAN DATASET TEST DEEPFRUITS
# ==========================================

def prepare_dataset():

    images_dir = OUTPUT_ROOT / "test" / "images"
    labels_dir = OUTPUT_ROOT / "test" / "labels"

    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    samples = [
        image for image in SOURCE_IMAGES.iterdir()
        if image.is_file()
        and image.stem.startswith("deepfruits_")
        and image.suffix.lower() in {
            ".jpg", ".jpeg", ".png"
        }
    ]

    if len(samples) != 18:
        raise ValueError(
            f"Jumlah gambar DeepFruits tidak sesuai: {len(samples)}"
        )

    for image in samples:

        label = SOURCE_LABELS / f"{image.stem}.txt"

        if not label.exists():
            raise FileNotFoundError(label)

        shutil.copy2(
            image,
            images_dir / image.name
        )

        shutil.copy2(
            label,
            labels_dir / label.name
        )

    yaml_content = f"""path: {OUTPUT_ROOT.as_posix()}
train: test/images
val: test/images
test: test/images

nc: 1
names: ['mango']
"""

    DATA_YAML.write_text(yaml_content)

    print("Dataset DeepFruits berhasil disiapkan.")
    print("Jumlah gambar:", len(samples))


# ==========================================
# EVALUASI MODEL
# ==========================================

def evaluate_model(model_name, model_path):

    print("\n" + "=" * 55)
    print("MENGUJI:", model_name)
    print("=" * 55)

    if not model_path.exists():
        raise FileNotFoundError(model_path)

    model = YOLO(str(model_path))

    result = model.val(
        data=str(DATA_YAML),
        split="test",
        imgsz=512,
        batch=2,
        device="cpu",
        workers=0,
        plots=True,
        project=str(RESULTS_DIR),
        name=model_name,
        exist_ok=True
    )

    return {
        "model": model_name,
        "precision": result.box.mp,
        "recall": result.box.mr,
        "map50": result.box.map50,
        "map50_95": result.box.map
    }


# ==========================================
# FUNGSI UTAMA
# ==========================================

def main():

    print("=" * 60)
    print("PERBANDINGAN MODEL PADA DEEPFRUITS")
    print("=" * 60)

    prepare_dataset()

    results = []

    results.append(
        evaluate_model(
            "model_gabungan",
            MODEL_GABUNGAN
        )
    )

    results.append(
        evaluate_model(
            "model_mangoyolo",
            MODEL_MANGOYOLO
        )
    )

    print("\n" + "=" * 85)
    print("HASIL PERBANDINGAN DEEPFRUITS")
    print("=" * 85)

    print(
        f"{'MODEL':<22}"
        f"{'PRECISION':<15}"
        f"{'RECALL':<15}"
        f"{'MAP50':<15}"
        f"{'MAP50-95':<15}"
    )

    print("-" * 85)

    for item in results:

        print(
            f"{item['model']:<22}"
            f"{item['precision']:<15.4f}"
            f"{item['recall']:<15.4f}"
            f"{item['map50']:<15.4f}"
            f"{item['map50_95']:<15.4f}"
        )

    print("\nEvaluasi selesai.")


if __name__ == "__main__":
    main()