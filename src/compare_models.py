from pathlib import Path
from ultralytics import YOLO


# ==========================================
# KONFIGURASI
# ==========================================

ROOT = Path(__file__).resolve().parents[1]

DATA_YAML = Path(
    "/run/media/mchmdzaki/4A688C98688C8483/"
    "UTS_Pengolahan_Citra_Data/dataset/raw/"
    "mango_detection_australia/data/yolo_converted/data.yaml"
)

MODELS = {
    "Model Gabungan": ROOT / "models" / "best_mango.pt",
    "Model MangoYOLO": ROOT / "models" / "best_mangoyolo_only.pt"
}

OUTPUT_DIR = ROOT / "runs" / "model_comparison"


# ==========================================
# EVALUASI KEDUA MODEL
# ==========================================

def main():

    print("=" * 65)
    print("PERBANDINGAN MODEL YOLO11N")
    print("DATA TESTING MANGOYOLO AUSTRALIA")
    print("=" * 65)

    if not DATA_YAML.exists():
        raise FileNotFoundError(DATA_YAML)

    results_summary = []

    for model_name, model_path in MODELS.items():

        print("\n" + "=" * 50)
        print("Menguji:", model_name)
        print("=" * 50)

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
            project=str(OUTPUT_DIR),
            name=model_name.lower().replace(" ", "_"),
            exist_ok=True
        )

        metrics = {
            "model": model_name,
            "precision": result.box.mp,
            "recall": result.box.mr,
            "map50": result.box.map50,
            "map50_95": result.box.map
        }

        results_summary.append(metrics)

    # ======================================
    # TAMPILKAN PERBANDINGAN
    # ======================================

    print("\n" + "=" * 85)
    print("RINGKASAN PERBANDINGAN")
    print("=" * 85)

    print(
        f"{'MODEL':<22}"
        f"{'PRECISION':<15}"
        f"{'RECALL':<15}"
        f"{'MAP50':<15}"
        f"{'MAP50-95':<15}"
    )

    print("-" * 85)

    for item in results_summary:

        print(
            f"{item['model']:<22}"
            f"{item['precision']:<15.4f}"
            f"{item['recall']:<15.4f}"
            f"{item['map50']:<15.4f}"
            f"{item['map50_95']:<15.4f}"
        )

    print("\nHasil evaluasi tersimpan di:")
    print(OUTPUT_DIR)


if __name__ == "__main__":
    main()