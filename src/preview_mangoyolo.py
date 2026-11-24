from pathlib import Path
import random
import math
import cv2


# Lokasi dataset hasil konversi
DATASET_DIR = Path(
    "/run/media/mchmdzaki/4A688C98688C8483/"
    "UTS_Pengolahan_Citra_Data/dataset/raw/"
    "mango_detection_australia/data/yolo_converted"
)

OUTPUT_DIR = (
    Path(__file__).resolve().parents[1]
    / "outputs"
    / "dataset_review"
)

IMAGE_DIR = DATASET_DIR / "train" / "images"
LABEL_DIR = DATASET_DIR / "train" / "labels"


def draw_yolo_boxes(image, label_path):

    height, width = image.shape[:2]
    count = 0

    if not label_path.exists():
        return image, count

    with open(label_path, "r") as file:

        for line in file:

            values = line.strip().split()

            if len(values) != 5:
                continue

            class_id = int(values[0])

            x_center, y_center, box_width, box_height = (
                map(float, values[1:])
            )

            x1 = int((x_center - box_width / 2) * width)
            y1 = int((y_center - box_height / 2) * height)

            x2 = int((x_center + box_width / 2) * width)
            y2 = int((y_center + box_height / 2) * height)

            cv2.rectangle(
                image,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            cv2.putText(
                image,
                "Mango",
                (x1, max(y1 - 5, 15)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2
            )

            count += 1

    return image, count


def main():

    print("=" * 45)
    print("PREVIEW DATASET MANGOYOLO")
    print("=" * 45)

    images = [
        p for p in IMAGE_DIR.iterdir()
        if p.suffix.lower() in {
            ".jpg", ".jpeg", ".png"
        }
    ]

    if not images:
        print("Gambar tidak ditemukan.")
        return

    random.seed(42)

    samples = random.sample(
        images,
        min(8, len(images))
    )

    columns = 2
    rows = math.ceil(len(samples) / columns)

    cell_width = 480
    cell_height = 370

    canvas = (
        255 * (
            0 * cv2.UMat(
                rows * cell_height,
                columns * cell_width,
                cv2.CV_8UC3
            ).get()
        )
    )

    canvas[:] = (35, 35, 35)

    for index, image_path in enumerate(samples):

        label_path = (
            LABEL_DIR / f"{image_path.stem}.txt"
        )

        image = cv2.imread(str(image_path))

        if image is None:
            continue

        image, count = draw_yolo_boxes(
            image,
            label_path
        )

        h, w = image.shape[:2]

        scale = min(
            450 / w,
            310 / h
        )

        new_w = int(w * scale)
        new_h = int(h * scale)

        image = cv2.resize(
            image,
            (new_w, new_h)
        )

        row = index // columns
        col = index % columns

        x = col * cell_width + (cell_width - new_w) // 2
        y = row * cell_height + 10

        canvas[y:y + new_h, x:x + new_w] = image

        cv2.putText(
            canvas,
            image_path.name,
            (col * cell_width + 10, row * cell_height + 340),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (255, 255, 255),
            1
        )

        cv2.putText(
            canvas,
            f"Jumlah buah: {count}",
            (col * cell_width + 10, row * cell_height + 360),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    output_path = OUTPUT_DIR / "mangoyolo_preview.jpg"

    cv2.imwrite(
        str(output_path),
        canvas
    )

    print("\nPreview berhasil dibuat!")
    print("Lokasi:", output_path)


if __name__ == "__main__":
    main()