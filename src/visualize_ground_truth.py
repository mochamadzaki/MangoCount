from pathlib import Path
import cv2
from ultralytics import YOLO


# ==========================================
# KONFIGURASI
# ==========================================

ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = ROOT / "models" / "best_mango.pt"

DATASET = Path(
    "/run/media/mchmdzaki/4A688C98688C8483/"
    "UTS_Pengolahan_Citra_Data/dataset/"
    "processed/mango_combined_final"
)

IMAGE_NAME = (
    "deepfruits_n12761284_7172_png.rf."
    "e1fe7b3a6412029d0d3e3f695a0ad1b0.jpg"
)

IMAGE_PATH = DATASET / "test" / "images" / IMAGE_NAME

LABEL_PATH = DATASET / "test" / "labels" / (
    Path(IMAGE_NAME).stem + ".txt"
)

OUTPUT_DIR = ROOT / "outputs" / "dataset_review"


# ==========================================
# GAMBAR GROUND TRUTH
# ==========================================

def draw_ground_truth(image, label_path):

    height, width = image.shape[:2]
    count = 0

    for line in label_path.read_text().splitlines():

        if not line.strip():
            continue

        values = line.split()

        class_id = int(values[0])

        x, y, w, h = map(float, values[1:])

        x1 = int((x - w / 2) * width)
        y1 = int((y - h / 2) * height)

        x2 = int((x + w / 2) * width)
        y2 = int((y + h / 2) * height)

        cv2.rectangle(
            image,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )

        cv2.putText(
            image,
            "GT Mango",
            (x1, max(y1 - 5, 15)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2
        )

        count += 1

    return image, count


# ==========================================
# GAMBAR PREDICTION
# ==========================================

def draw_prediction(image, result):

    count = 0

    for box in result.boxes:

        x1, y1, x2, y2 = (
            box.xyxy[0].cpu().numpy().astype(int)
        )

        confidence = float(box.conf[0])

        cv2.rectangle(
            image,
            (x1, y1),
            (x2, y2),
            (0, 0, 255),
            2
        )

        cv2.putText(
            image,
            f"Mango {confidence:.2f}",
            (x1, max(y1 - 5, 15)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 255),
            2
        )

        count += 1

    return image, count


# ==========================================
# FUNGSI UTAMA
# ==========================================

def main():

    print("=" * 50)
    print("GROUND TRUTH VS PREDICTION")
    print("=" * 50)

    if not IMAGE_PATH.exists():
        raise FileNotFoundError(IMAGE_PATH)

    if not LABEL_PATH.exists():
        raise FileNotFoundError(LABEL_PATH)

    image = cv2.imread(str(IMAGE_PATH))

    if image is None:
        raise ValueError("Gambar gagal dibaca.")

    # Ground Truth
    gt_image, gt_count = draw_ground_truth(
        image.copy(),
        LABEL_PATH
    )

    # Memuat model
    model = YOLO(str(MODEL_PATH))

    # Prediction
    result = model.predict(
        source=str(IMAGE_PATH),
        imgsz=512,
        conf=0.05,
        device="cpu",
        verbose=False
    )[0]

    pred_image, pred_count = draw_prediction(
        image.copy(),
        result
    )

    # Judul gambar
    cv2.putText(
        gt_image,
        f"GROUND TRUTH: {gt_count}",
        (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2
    )

    cv2.putText(
        pred_image,
        f"PREDICTION: {pred_count}",
        (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 0, 255),
        2
    )

    # Gabungkan berdampingan
    comparison = cv2.hconcat([
        gt_image,
        pred_image
    ])

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    output = OUTPUT_DIR / "ground_truth_comparison.jpg"

    cv2.imwrite(
        str(output),
        comparison
    )

    print("\nGround Truth:", gt_count)
    print("Prediction  :", pred_count)
    print("Selisih     :", pred_count - gt_count)

    print("\nPreview tersimpan:")
    print(output)


if __name__ == "__main__":
    main()