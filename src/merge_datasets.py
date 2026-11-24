from pathlib import Path
from collections import Counter
import hashlib
import math
import shutil


# =====================================================
# 1. KONFIGURASI DATASET
# =====================================================

RAW_DIR = Path(
    "/run/media/mchmdzaki/4A688C98688C8483/"
    "UTS_Pengolahan_Citra_Data/dataset/raw"
)

# Dataset utama MangoYOLO
MANGOYOLO_DIR = (
    RAW_DIR
    / "mango_detection_australia"
    / "data"
    / "yolo_converted"
)

# Dataset pendamping DeepFruits
DEEPFRUITS_DIR = RAW_DIR / "deepfruits_mango_v7"

# Hasil akhir disimpan di folder baru
OUTPUT_DIR = (
    RAW_DIR.parent
    / "processed"
    / "mango_combined_final"
)

# Satu kelas deteksi
CLASS_ID = 0
CLASS_NAME = "mango"

SPLITS = ["train", "valid", "test"]

SOURCES = [
    ("mangoyolo", MANGOYOLO_DIR),
    ("deepfruits", DEEPFRUITS_DIR),
]

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp"
}


# =====================================================
# 2. STATISTIK
# =====================================================

statistics = {
    split: {
        "images": 0,
        "objects": 0,
        "invalid_boxes": 0,
        "duplicates": 0
    }
    for split in SPLITS
}

source_statistics = Counter()

# Mencatat hash gambar untuk deteksi duplikasi
hash_registry = {}


# =====================================================
# 3. MENGHITUNG HASH GAMBAR
# =====================================================

def calculate_hash(image_path):

    sha256 = hashlib.sha256()

    with open(image_path, "rb") as file:

        for chunk in iter(
            lambda: file.read(1024 * 1024),
            b""
        ):
            sha256.update(chunk)

    return sha256.hexdigest()


# =====================================================
# 4. MENGAMBIL FILE GAMBAR DAN LABEL
# =====================================================

def collect_dataset(source_dir, split):

    image_dir = source_dir / split / "images"
    label_dir = source_dir / split / "labels"

    if not image_dir.exists():
        raise FileNotFoundError(
            f"Folder gambar tidak ditemukan: {image_dir}"
        )

    if not label_dir.exists():
        raise FileNotFoundError(
            f"Folder label tidak ditemukan: {label_dir}"
        )

    images = [
        file
        for file in image_dir.iterdir()
        if file.is_file()
        and file.suffix.lower() in IMAGE_EXTENSIONS
    ]

    labels = list(label_dir.glob("*.txt"))

    image_map = {}

    for image in images:

        if image.stem in image_map:
            raise ValueError(
                f"Nama gambar duplikat: {image.stem}"
            )

        image_map[image.stem] = image

    label_map = {
        label.stem: label
        for label in labels
    }

    missing_labels = (
        set(image_map.keys())
        - set(label_map.keys())
    )

    extra_labels = (
        set(label_map.keys())
        - set(image_map.keys())
    )

    if missing_labels or extra_labels:

        raise ValueError(
            f"Ketidakcocokan gambar dan label pada "
            f"{source_dir.name}/{split}\n"
            f"Gambar tanpa label: {len(missing_labels)}\n"
            f"Label tanpa gambar: {len(extra_labels)}"
        )

    return [
        (image_map[stem], label_map[stem])
        for stem in sorted(image_map)
    ]


# =====================================================
# 5. VALIDASI ANOTASI YOLO
# =====================================================

def validate_label(label_path):

    valid_lines = []
    invalid_count = 0

    lines = label_path.read_text().splitlines()

    for line_number, line in enumerate(lines, 1):

        if not line.strip():
            continue

        values = line.split()

        if len(values) != 5:

            raise ValueError(
                f"Format anotasi salah: {label_path}, "
                f"baris {line_number}"
            )

        try:

            class_id = int(values[0])

            x = float(values[1])
            y = float(values[2])
            w = float(values[3])
            h = float(values[4])

        except ValueError:

            raise ValueError(
                f"Nilai anotasi tidak valid: {label_path}, "
                f"baris {line_number}"
            )

        # Periksa kelas
        if class_id != CLASS_ID:

            raise ValueError(
                f"ID kelas tidak sesuai: {label_path}"
            )

        coordinates = [x, y, w, h]

        # Pastikan semua angka valid
        if not all(
            math.isfinite(value)
            for value in coordinates
        ):

            raise ValueError(
                f"Koordinat tidak valid: {label_path}"
            )

        # Periksa koordinat pusat
        if not (0 <= x <= 1 and 0 <= y <= 1):

            raise ValueError(
                f"Posisi bounding box di luar gambar: "
                f"{label_path}"
            )

        # Periksa ukuran bounding box
        if w < 0 or h < 0 or w > 1 or h > 1:

            raise ValueError(
                f"Ukuran bounding box tidak valid: "
                f"{label_path}"
            )

        # Abaikan bounding box tanpa luas
        if w == 0 or h == 0:

            invalid_count += 1
            continue

        # Pertahankan anotasi valid
        valid_lines.append(
            f"{CLASS_ID} "
            f"{x:.8f} "
            f"{y:.8f} "
            f"{w:.8f} "
            f"{h:.8f}"
        )

    return valid_lines, invalid_count


# =====================================================
# 6. MEMBUAT FOLDER OUTPUT
# =====================================================

def prepare_output():

    if OUTPUT_DIR.exists() and any(
        OUTPUT_DIR.iterdir()
    ):

        raise FileExistsError(
            "\nFolder output sudah berisi data.\n"
            "Program dihentikan untuk menghindari "
            "penimpaan data."
            f"\nLokasi: {OUTPUT_DIR}"
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )


# =====================================================
# 7. MENYALIN DAN MENGGABUNGKAN DATASET
# =====================================================

def merge_split(split):

    print(f"\n{'=' * 50}")
    print(f"PROSES {split.upper()}")
    print(f"{'=' * 50}")

    for source_name, source_dir in SOURCES:

        print(f"\nSumber: {source_name}")

        dataset = collect_dataset(
            source_dir,
            split
        )

        print("Jumlah gambar:", len(dataset))

        for image_path, label_path in dataset:

            # Validasi anotasi
            valid_lines, invalid_count = validate_label(
                label_path
            )

            object_count = len(valid_lines)

            # Hitung hash gambar
            digest = calculate_hash(image_path)

            # Periksa duplikasi
            if digest in hash_registry:

                previous = hash_registry[digest]

                # Duplikat lintas split berisiko menyebabkan
                # kebocoran data (data leakage)
                if previous["split"] != split:

                    raise ValueError(
                        "\nDuplikasi gambar ditemukan "
                        "pada split berbeda!\n"
                        f"Gambar: {image_path.name}\n"
                        f"Split sebelumnya: {previous['split']}\n"
                        f"Split sekarang: {split}"
                    )

                # Periksa kesamaan anotasi
                if previous["labels"] != valid_lines:

                    raise ValueError(
                        "\nGambar identik memiliki anotasi berbeda!\n"
                        f"Gambar: {image_path.name}"
                    )

                statistics[split]["duplicates"] += 1
                continue

            hash_registry[digest] = {
                "split": split,
                "labels": valid_lines
            }

            # Nama baru untuk menghindari bentrok
            new_stem = (
                f"{source_name}_{image_path.stem}"
            )

            # Lokasi gambar output
            output_image = (
                OUTPUT_DIR
                / split
                / "images"
                / f"{new_stem}{image_path.suffix.lower()}"
            )

            # Lokasi label output
            output_label = (
                OUTPUT_DIR
                / split
                / "labels"
                / f"{new_stem}.txt"
            )

            output_image.parent.mkdir(
                parents=True,
                exist_ok=True
            )

            output_label.parent.mkdir(
                parents=True,
                exist_ok=True
            )

            # Salin gambar
            shutil.copy2(
                image_path,
                output_image
            )

            # Simpan label yang sudah dibersihkan
            output_label.write_text(
                "\n".join(valid_lines)
            )

            # Perbarui statistik
            statistics[split]["images"] += 1
            statistics[split]["objects"] += object_count
            statistics[split]["invalid_boxes"] += invalid_count

            source_statistics[source_name] += 1

        print(
            f"Berhasil memproses {len(dataset)} gambar."
        )


# =====================================================
# 8. MEMBUAT DATA.YAML
# =====================================================

def create_data_yaml():

    content = f"""path: {OUTPUT_DIR.as_posix()}

train: train/images
val: valid/images
test: test/images

nc: 1
names: ['{CLASS_NAME}']
"""

    yaml_path = OUTPUT_DIR / "data.yaml"

    yaml_path.write_text(content)

    print("\ndata.yaml berhasil dibuat.")


# =====================================================
# 9. MEMBUAT LAPORAN PENGGABUNGAN
# =====================================================

def create_report():

    report_path = OUTPUT_DIR / "merge_report.txt"

    lines = [
        "LAPORAN PENGGABUNGAN DATASET MANGGA",
        "=" * 45,
        "",
        "SUMBER DATASET:",
        "1. MangoYOLO Australia",
        "2. DeepFruits Mango",
        "",
        "HASIL PENGGABUNGAN:"
    ]

    total_images = 0
    total_objects = 0
    total_invalid = 0
    total_duplicates = 0

    for split in SPLITS:

        data = statistics[split]

        lines.append(f"\n{split.upper()}")
        lines.append(f"Gambar: {data['images']}")
        lines.append(f"Objek: {data['objects']}")
        lines.append(f"Bounding box diabaikan: {data['invalid_boxes']}")
        lines.append(f"Duplikat: {data['duplicates']}")

        total_images += data["images"]
        total_objects += data["objects"]
        total_invalid += data["invalid_boxes"]
        total_duplicates += data["duplicates"]

    lines.extend([
        "",
        "=" * 45,
        "TOTAL DATASET",
        f"Total gambar: {total_images}",
        f"Total objek: {total_objects}",
        f"Bounding box diabaikan: {total_invalid}",
        f"Duplikat dilewati: {total_duplicates}",
        "",
        "DISTRIBUSI SUMBER:"
    ])

    for source, count in source_statistics.items():
        lines.append(f"{source}: {count} gambar")

    report_path.write_text(
        "\n".join(lines)
    )

    print("\n".join(lines))


# =====================================================
# 10. FUNGSI UTAMA
# =====================================================

def main():

    print("=" * 55)
    print("PENGGABUNGAN DATASET MANGGA")
    print("=" * 55)

    # Periksa sumber dataset
    for source_name, source_dir in SOURCES:

        if not source_dir.exists():

            raise FileNotFoundError(
                f"Dataset tidak ditemukan: {source_name}"
            )

    # Siapkan output baru
    prepare_output()

    # Proses seluruh split
    for split in SPLITS:

        merge_split(split)

    # Buat konfigurasi YOLO
    create_data_yaml()

    # Buat laporan
    create_report()

    print("\n" + "=" * 55)
    print("PENGGABUNGAN BERHASIL!")
    print("=" * 55)

    print("\nLokasi dataset final:")
    print(OUTPUT_DIR)

    print("\nDataset siap untuk pemeriksaan visual.")


# =====================================================
# EKSEKUSI
# =====================================================

if __name__ == "__main__":

    try:
        main()

    except Exception as error:

        print("\n" + "=" * 55)
        print("PROSES PENGGABUNGAN GAGAL")
        print("=" * 55)

        print("Jenis error:", type(error).__name__)
        print("Pesan:", error)

        raise