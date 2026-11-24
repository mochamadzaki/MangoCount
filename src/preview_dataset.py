from pathlib import Path
import random
import cv2
import math


DATA_ROOT = Path(
    "/run/media/mchmdzaki/4A688C98688C8483/"
    "UTS_Pengolahan_Citra_Data/dataset/raw"
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "dataset_review"

DATASETS = {
    "DeepFruits Mango": DATA_ROOT / "deepfruits_mango_v7",
    "Mango Fruits Detection": DATA_ROOT / "mango_fruits_detection_v1",
}

IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".webp"
}


def get_images(dataset_path):
    images = []

    for split in ["train", "valid", "test"]:
        image_dir = dataset_path / split / "images"
        label_dir = dataset_path / split / "labels"

        if not image_dir.exists():
            continue

        for image_path in image_dir.iterdir():

            if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue

            label_path = label_dir / f"{image_path.stem}.txt"

            images.append((image_path, label_path))

    return images


def draw_annotations(image, label_path):

    height, width = image.shape[:2]
    count = 0

    if not label_path.exists():
        return image, count

    with open(label_path, "r") as file:

        for line in file:

            values = line.strip().split()

            if len(values) != 5:
                continue

            try:
                class_id = int(float(values[0]))

                x_center = float(values[1]) * width
                y_center = float(values[2]) * height
                box_width = float(values[3]) * width
                box_height = float(values[4]) * height

                x1 = int(x_center - box_width / 2)
                y1 = int(y_center - box_height / 2)

                x2 = int(x_center + box_width / 2)
                y2 = int(y_center + box_height / 2)

                x1 = max(0, min(x1, width - 1))
                y1 = max(0, min(y1, height - 1))
                x2 = max(0, min(x2, width - 1))
                y2 = max(0, min(y2, height - 1))

                cv2.rectangle(
                    image,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    2
                )

                cv2.putText(
                    image,
                    f"Mango ({class_id})",
                    (x1, max(y1 - 5, 15)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2
                )

                count += 1

            except (ValueError, OverflowError):
                continue

    return image, count


def create_preview(dataset_name, dataset_path):

    print(f"\nMemeriksa: {dataset_name}")

    images = get_images(dataset_path)

    if not images:
        print("Tidak ada gambar ditemukan.")
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

    preview = 40 * (
        rows * cell_height
    )

    canvas = (
        255 * (
            0 * (
                cv2.UMat(1, 1, cv2.CV_8UC3)
                .get()
            )
        )
    )

    sheet = cv2.UMat(
        rows * cell_height,
        columns * cell_width,
        cv2.CV_8UC3
    ).get()

    sheet[:] = (35, 35, 35)

    for index, (image_path, label_path) in enumerate(samples):

        image = cv2.imread(str(image_path))

        if image is None:
            continue

        image, count = draw_annotations(
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

        sheet[y:y + new_h, x:x + new_w] = image

        cv2.putText(
            sheet,
            image_path.name[:45],
            (col * cell_width + 10, row * cell_height + 340),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1
        )

        cv2.putText(
            sheet,
            f"Jumlah bounding box: {count}",
            (col * cell_width + 10, row * cell_height + 360),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 255, 0),
            1
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    safe_name = dataset_name.lower().replace(" ", "_")

    output_path = OUTPUT_DIR / f"{safe_name}.jpg"

    cv2.imwrite(
        str(output_path),
        sheet
    )

    print("Preview tersimpan:", output_path)


def main():

    print("=" * 50)
    print("PEMERIKSAAN VISUAL DATASET MANGGA")
    print("=" * 50)

    for name, path in DATASETS.items():

        if not path.exists():
            print(f"\nDataset tidak ditemukan: {name}")
            continue

        create_preview(name, path)

    print("\nPemeriksaan selesai.")


if __name__ == "__main__":
    main()