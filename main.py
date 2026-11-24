from pathlib import Path
from getpass import getpass
from roboflow import Roboflow


# ==================================================
# KONFIGURASI DATASET
# ==================================================

DATA_ROOT = Path(
    "/run/media/mchmdzaki/4A688C98688C8483"
)

DATASET_PARENT = (
    DATA_ROOT
    / "UTS_Pengolahan_Citra_Data"
    / "dataset"
    / "raw"
)

DATASET_DIR = DATASET_PARENT / "mango_fruits_detection_v1"

WORKSPACE_NAME = "mango-fruit-detection-sb96u"
PROJECT_NAME = "mango-fruits-detection"
VERSION_NUMBER = 1

DATASET_FORMAT = "yolov11"


# ==================================================
# PEMERIKSAAN DATASET
# ==================================================

def check_dataset():

    print("\n===== PEMERIKSAAN DATASET =====")

    if not DATASET_DIR.exists():
        print("Folder dataset tidak ditemukan.")
        return

    extensions = {".jpg", ".jpeg", ".png", ".webp"}

    for split in ["train", "valid", "test"]:

        image_dir = DATASET_DIR / split / "images"
        label_dir = DATASET_DIR / split / "labels"

        images = (
            [f for f in image_dir.rglob("*")
             if f.is_file() and f.suffix.lower() in extensions]
            if image_dir.exists() else []
        )

        labels = (
            list(label_dir.rglob("*.txt"))
            if label_dir.exists() else []
        )

        print(f"\n{split.upper()}")
        print("Jumlah gambar  :", len(images))
        print("Jumlah anotasi :", len(labels))

    yaml_file = DATASET_DIR / "data.yaml"

    print("\n===== DATA.YAML =====")

    if yaml_file.exists():
        print(yaml_file.read_text())
    else:
        print("data.yaml tidak ditemukan.")


# ==================================================
# DOWNLOAD DATASET
# ==================================================

def main():

    print("=" * 50)
    print("DOWNLOAD MANGO FRUITS DETECTION")
    print("=" * 50)

    # Periksa drive data
    if not DATA_ROOT.exists():
        print("ERROR: Drive penyimpanan tidak ditemukan.")
        return

    # Meminta Private API Key
    api_key = getpass(
        "\nMasukkan Private API Key Roboflow: "
    ).strip()

    if not api_key:
        print("API Key tidak boleh kosong.")
        return

    try:

        print("\nMenghubungkan ke Roboflow...")

        rf = Roboflow(api_key=api_key)

        print("Koneksi berhasil!")

        # Mengakses workspace
        print("\nMengakses workspace...")

        workspace = rf.workspace(WORKSPACE_NAME)

        # Mengakses project
        project = workspace.project(PROJECT_NAME)

        print("Project berhasil ditemukan.")

        # Mengakses versi dataset
        version = project.version(VERSION_NUMBER)

        print("\nInformasi dataset:")
        print("Project :", PROJECT_NAME)
        print("Version :", VERSION_NUMBER)
        print("Format  :", DATASET_FORMAT)
        print("Lokasi  :", DATASET_DIR)

        # Membuat folder induk.
        # Folder dataset akan dibuat oleh Roboflow SDK.
        DATASET_PARENT.mkdir(
            parents=True,
            exist_ok=True
        )

        # Mengunduh dataset
        print("\nMulai mengunduh dataset...")
        print("Mohon tunggu sampai selesai.\n")

        dataset = version.download(
            model_format=DATASET_FORMAT,
            location=str(DATASET_DIR),
            overwrite=True
        )

        print("\n" + "=" * 50)
        print("DOWNLOAD SELESAI!")
        print("=" * 50)

        print("Lokasi:", dataset.location)

        # Memeriksa hasil unduhan
        check_dataset()

    except Exception as error:

        print("\n" + "=" * 50)
        print("DOWNLOAD GAGAL!")
        print("=" * 50)

        print("Jenis error:", type(error).__name__)
        print("Pesan:", error)


# ==================================================
# EKSEKUSI PROGRAM
# ==================================================

if __name__ == "__main__":
    main()