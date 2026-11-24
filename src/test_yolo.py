from pathlib import Path
from ultralytics import YOLO


# ==========================================
# KONFIGURASI
# ==========================================

PROJECT_DIR = Path(__file__).resolve().parents[1]

MODEL_PATH = PROJECT_DIR / "models" / "best_mango.pt"

DATA_YAML = Path(
    "/run/media/mchmdzaki/4A688C98688C8483/"
    "UTS_Pengolahan_Citra_Data/dataset/"
    "processed/mango_combined_final/data.yaml"
)

OUTPUT_DIR = PROJECT_DIR / "runs" / "mango_evaluation"


# ==========================================
# EVALUASI DATA TESTING
# ==========================================

def main():

    print("=" * 50)
    print("EVALUASI MODEL YOLO11N - DATA TESTING")
    print("=" * 50)

    if not MODEL_PATH.exists():
        raise FileNotFoundError(MODEL_PATH)

    if not DATA_YAML.exists():
        raise FileNotFoundError(DATA_YAML)

    print("\nMemuat model:", MODEL_PATH)

    model = YOLO(str(MODEL_PATH))

    print("\nMemulai evaluasi pada 318 gambar testing...")

    results = model.val(
        data=str(DATA_YAML),
        split="test",
        imgsz=512,
        batch=2,
        device="cpu",
        workers=0,
        plots=True,
        project=str(OUTPUT_DIR),
        name="mango_test_final",
        exist_ok=False
    )

    print("\n" + "=" * 50)
    print("HASIL EVALUASI DATA TESTING")
    print("=" * 50)

    print(f"Precision  : {results.box.mp:.4f}")
    print(f"Recall     : {results.box.mr:.4f}")
    print(f"mAP50      : {results.box.map50:.4f}")
    print(f"mAP50-95   : {results.box.map:.4f}")

    print("\nLokasi hasil:")
    print(OUTPUT_DIR / "mango_test_final")


if __name__ == "__main__":
    main()