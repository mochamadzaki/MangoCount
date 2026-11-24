from pathlib import Path
from ultralytics import YOLO


# ==================================
# KONFIGURASI
# ==================================

ROOT = Path(__file__).resolve().parents[1]

MODEL = ROOT / "models" / "best_mango.pt"

IMAGE = Path(
    "/run/media/mchmdzaki/4A688C98688C8483/"
    "UTS_Pengolahan_Citra_Data/dataset/"
    "processed/mango_combined_final/test/images/"
    "deepfruits_n12761284_7172_png.rf."
    "e1fe7b3a6412029d0d3e3f695a0ad1b0.jpg"
)

OUTPUT = ROOT / "runs" / "resolution_test"

IMAGE_SIZES = [512, 640, 768]

CONFIDENCE = 0.05


# ==================================
# PENGUJIAN RESOLUSI
# ==================================

def main():

    print("=" * 50)
    print("PENGUJIAN RESOLUSI YOLO11N")
    print("=" * 50)

    model = YOLO(str(MODEL))

    for size in IMAGE_SIZES:

        print(f"\nPengujian Image Size: {size}")

        results = model.predict(
            source=str(IMAGE),
            imgsz=size,
            conf=CONFIDENCE,
            device="cpu",
            save=True,
            project=str(OUTPUT),
            name=f"imgsz_{size}",
            exist_ok=True,
            verbose=False
        )

        result = results[0]

        print("Jumlah deteksi:", len(result.boxes))

        scores = result.boxes.conf.tolist()

        print(
            "Confidence:",
            [round(score, 3) for score in scores]
        )

    print("\nPengujian selesai.")
    print("Lokasi hasil:", OUTPUT)


if __name__ == "__main__":
    main()