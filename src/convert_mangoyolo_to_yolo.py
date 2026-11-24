from pathlib import Path
import xml.etree.ElementTree as ET
import shutil
import cv2
import math


# ==================================================
# KONFIGURASI
# ==================================================

DATA_ROOT = Path(
    "/run/media/mchmdzaki/4A688C98688C8483/"
    "UTS_Pengolahan_Citra_Data/dataset/raw/"
    "mango_detection_australia"
)

VOC_DIR = (
    DATA_ROOT
    / "data/origin/extracted/VOCDevkit/VOC2007"
)

IMAGE_DIR = VOC_DIR / "JPEGImages"
ANNOTATION_DIR = VOC_DIR / "Annotations"
SPLIT_DIR = VOC_DIR / "ImageSets/Main"

OUTPUT_DIR = DATA_ROOT / "data/yolo_converted"

SPLITS = {
    "train": "train",
    "val": "valid",
    "test": "test"
}

CLASS_ID = 0
CLASS_NAME = "mango"
SOURCE_CLASS = "M"


# ==================================================
# MEMBACA DATASET
# ==================================================

def get_image_map():

    image_map = {}

    extensions = {".jpg", ".jpeg", ".png"}

    for image in IMAGE_DIR.iterdir():

        if image.is_file() and image.suffix.lower() in extensions:

            image_map[image.stem] = image

    return image_map


def read_split(split_name):

    split_file = SPLIT_DIR / f"{split_name}.txt"

    if not split_file.exists():
        raise FileNotFoundError(
            f"File pembagian tidak ditemukan: {split_file}"
        )

    return [
        line.strip()
        for line in split_file.read_text().splitlines()
        if line.strip()
    ]


# ==================================================
# KONVERSI BOUNDING BOX
# ==================================================

def convert_bbox(obj, image_width, image_height):

    bbox = obj.find("bndbox")

    if bbox is None:
        return None

    xmin = float(bbox.findtext("xmin"))
    ymin = float(bbox.findtext("ymin"))
    xmax = float(bbox.findtext("xmax"))
    ymax = float(bbox.findtext("ymax"))

    # Batasi koordinat di dalam gambar
    xmin = max(0, min(xmin, image_width))
    ymin = max(0, min(ymin, image_height))

    xmax = max(0, min(xmax, image_width))
    ymax = max(0, min(ymax, image_height))

    if xmax <= xmin or ymax <= ymin:
        return None

    # Normalisasi koordinat YOLO
    x_center = ((xmin + xmax) / 2) / image_width
    y_center = ((ymin + ymax) / 2) / image_height

    box_width = (xmax - xmin) / image_width
    box_height = (ymax - ymin) / image_height

    values = [
        x_center,
        y_center,
        box_width,
        box_height
    ]

    if not all(math.isfinite(v) for v in values):
        return None

    return values


# ==================================================
# PROSES SATU GAMBAR
# ==================================================

def process_image(image_id, image_map, split_name):

    image_path = image_map.get(image_id)

    xml_path = ANNOTATION_DIR / f"{image_id}.xml"

    if image_path is None:
        raise FileNotFoundError(
            f"Gambar tidak ditemukan: {image_id}"
        )

    if not xml_path.exists():
        raise FileNotFoundError(
            f"XML tidak ditemukan: {image_id}"
        )

    # Baca gambar untuk mendapatkan ukuran sebenarnya
    image = cv2.imread(str(image_path))

    if image is None:
        raise ValueError(
            f"Gagal membaca gambar: {image_path}"
        )

    image_height, image_width = image.shape[:2]

    # Baca XML
    tree = ET.parse(xml_path)
    root = tree.getroot()

    label_lines = []
    invalid_count = 0

    for obj in root.findall("object"):

        class_name = obj.findtext(
            "name",
            default=""
        ).strip()

        if class_name != SOURCE_CLASS:
            raise ValueError(
                f"Kelas tidak sesuai: {class_name} "
                f"pada {xml_path.name}"
            )

        bbox_values = convert_bbox(
            obj,
            image_width,
            image_height
        )

        if bbox_values is None:
            invalid_count += 1
            continue

        x, y, w, h = bbox_values

        label_lines.append(
            f"{CLASS_ID} "
            f"{x:.6f} "
            f"{y:.6f} "
            f"{w:.6f} "
            f"{h:.6f}"
        )

    # Folder output
    image_output = (
        OUTPUT_DIR
        / split_name
        / "images"
        / image_path.name
    )

    label_output = (
        OUTPUT_DIR
        / split_name
        / "labels"
        / f"{image_id}.txt"
    )

    image_output.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    label_output.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    # Salin gambar
    shutil.copy2(
        image_path,
        image_output
    )

    # Simpan anotasi YOLO
    label_output.write_text(
        "\n".join(label_lines)
    )

    return len(label_lines), invalid_count


# ==================================================
# MEMBUAT DATA.YAML
# ==================================================

def create_yaml():

    yaml_content = f"""path: {OUTPUT_DIR.as_posix()}
train: train/images
val: valid/images
test: test/images

nc: 1
names: ['{CLASS_NAME}']
"""

    yaml_path = OUTPUT_DIR / "data.yaml"

    yaml_path.write_text(yaml_content)

    print("\ndata.yaml berhasil dibuat.")


# ==================================================
# FUNGSI UTAMA
# ==================================================

def main():

    print("=" * 55)
    print("KONVERSI MANGOYOLO XML KE YOLO11")
    print("=" * 55)

    # Pemeriksaan sumber
    if not IMAGE_DIR.exists():
        raise FileNotFoundError("Folder gambar tidak ditemukan.")

    if not ANNOTATION_DIR.exists():
        raise FileNotFoundError("Folder XML tidak ditemukan.")

    # Hindari menimpa hasil sebelumnya
    if OUTPUT_DIR.exists() and any(OUTPUT_DIR.iterdir()):

        print("\nFolder hasil konversi sudah berisi data.")
        print("Hentikan proses agar data sebelumnya tidak tertimpa.")
        return

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    image_map = get_image_map()

    total_images = 0
    total_objects = 0
    total_invalid = 0

    print("\nJumlah gambar sumber:", len(image_map))

    # Proses train, validation, test
    for source_split, output_split in SPLITS.items():

        print(f"\n===== {source_split.upper()} =====")

        image_ids = read_split(source_split)

        print("Jumlah data:", len(image_ids))

        for image_id in image_ids:

            object_count, invalid_count = process_image(
                image_id,
                image_map,
                output_split
            )

            total_images += 1
            total_objects += object_count
            total_invalid += invalid_count

        print("Selesai memproses:", len(image_ids))

    # Membuat konfigurasi
    create_yaml()

    print("\n" + "=" * 55)
    print("KONVERSI BERHASIL")
    print("=" * 55)

    print("Total gambar       :", total_images)
    print("Total objek mangga :", total_objects)
    print("Bounding box invalid:", total_invalid)

    print("\nLokasi hasil:")
    print(OUTPUT_DIR)

    print("\nStruktur:")
    print("data/yolo_converted/")
    print("├── train/")
    print("├── valid/")
    print("├── test/")
    print("└── data.yaml")


if __name__ == "__main__":
    main()