from pathlib import Path
from ultralytics import YOLO
import cv2
import csv
import argparse
import time


# ==================================================
# 1. KONFIGURASI
# ==================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = PROJECT_ROOT / "models" / "best_mango.pt"

TRACKER_CONFIG = (
    PROJECT_ROOT / "configs" / "bytetrack_mango.yaml"
)

DATA_ROOT = Path(
    "/run/media/mchmdzaki/4A688C98688C8483/"
    "UTS_Pengolahan_Citra_Data"
)

VIDEO_DIR = DATA_ROOT / "videos"

OUTPUT_DIR = VIDEO_DIR / "output"

# Konfigurasi YOLO
IMAGE_SIZE = 640
CONFIDENCE = 0.10
IOU_THRESHOLD = 0.50
MAX_DETECTIONS = 1000


# ==================================================
# 2. FUNGSI TRACKING VIDEO
# ==================================================

def track_video(video_path):

    video_path = Path(video_path)

    print("=" * 55)
    print("YOLO11N + BYTETRACK")
    print("=" * 55)

    # --------------------------------------------------
    # Pemeriksaan model
    # --------------------------------------------------

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model tidak ditemukan: {MODEL_PATH}"
        )

    # --------------------------------------------------
    # Pemeriksaan konfigurasi tracker
    # --------------------------------------------------

    if not TRACKER_CONFIG.exists():
        raise FileNotFoundError(
            f"Konfigurasi tracker tidak ditemukan: "
            f"{TRACKER_CONFIG}"
        )

    # --------------------------------------------------
    # Pemeriksaan video
    # --------------------------------------------------

    if not video_path.exists():
        raise FileNotFoundError(
            f"Video tidak ditemukan: {video_path}"
        )

    if video_path.suffix.lower() not in {
        ".mp4", ".avi", ".mov", ".mkv"
    }:
        raise ValueError(
            "Format video tidak didukung."
        )

    # --------------------------------------------------
    # Folder output
    # --------------------------------------------------

    output_dir = OUTPUT_DIR / video_path.stem

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    # Video hasil tracking
    output_video = (
        output_dir / f"{video_path.stem}_tracked.mp4"
    )

    # CSV detail setiap objek
    output_csv = (
        output_dir / f"{video_path.stem}_detections.csv"
    )

    # CSV ringkasan setiap frame
    output_frame_csv = (
        output_dir / f"{video_path.stem}_frames.csv"
    )

    # --------------------------------------------------
    # Membuka video
    # --------------------------------------------------

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise RuntimeError(
            "Video gagal dibuka."
        )

    fps = cap.get(cv2.CAP_PROP_FPS)

    width = int(
        cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    )

    height = int(
        cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    )

    total_frames = int(
        cap.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    if fps <= 0:
        fps = 25.0

    if width <= 0 or height <= 0:
        cap.release()

        raise ValueError(
            "Resolusi video tidak valid."
        )

    print("\nInformasi video:")
    print("Nama video :", video_path.name)
    print("Resolusi   :", width, "x", height)
    print("FPS        :", fps)
    print("Total frame:", total_frames)

    # --------------------------------------------------
    # Membuat video output
    # --------------------------------------------------

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    writer = cv2.VideoWriter(
        str(output_video),
        fourcc,
        fps,
        (width, height)
    )

    if not writer.isOpened():

        cap.release()

        raise RuntimeError(
            "Gagal membuat video hasil tracking."
        )

    try:

        # --------------------------------------------------
        # Memuat model
        # --------------------------------------------------

        print("\nMemuat model YOLO11n...")

        model = YOLO(str(MODEL_PATH))

        print("Model berhasil dimuat.")

        # --------------------------------------------------
        # Statistik tracking
        # --------------------------------------------------

        unique_track_ids = set()

        frame_number = 0
        total_detections = 0
        frames_with_tracking = 0

        start_time = time.time()

        print("\nMemulai tracking...\n")

        # ==================================================
        # 3. MEMBUKA DUA FILE CSV
        # ==================================================

        with open(
            output_csv,
            "w",
            newline="",
            encoding="utf-8"
        ) as detection_file, open(
            output_frame_csv,
            "w",
            newline="",
            encoding="utf-8"
        ) as frame_file:

            detection_writer = csv.writer(
                detection_file
            )

            frame_writer = csv.writer(
                frame_file
            )

            # --------------------------------------------------
            # Header CSV DETAIL OBJEK
            # --------------------------------------------------

            detection_writer.writerow([
                "frame",
                "timestamp_seconds",
                "track_id",
                "class_id",
                "class_name",
                "confidence",
                "x1",
                "y1",
                "x2",
                "y2",
                "width",
                "height"
            ])

            # --------------------------------------------------
            # Header CSV RINGKASAN FRAME
            # --------------------------------------------------

            frame_writer.writerow([
                "frame",
                "timestamp_seconds",
                "detection_count",
                "track_ids"
            ])

            # ==================================================
            # 4. PROSES SETIAP FRAME VIDEO
            # ==================================================

            while True:

                success, frame = cap.read()

                if not success:
                    break

                frame_number += 1

                # --------------------------------------------------
                # YOLO + BYTETRACK
                # --------------------------------------------------

                results = model.track(
                    frame,
                    persist=True,
                    tracker=str(TRACKER_CONFIG),
                    imgsz=IMAGE_SIZE,
                    conf=CONFIDENCE,
                    iou=IOU_THRESHOLD,
                    max_det=MAX_DETECTIONS,
                    device="cpu",
                    verbose=False
                )

                result = results[0]

                boxes = result.boxes

                detection_count = (
                    len(boxes)
                    if boxes is not None
                    else 0
                )

                total_detections += detection_count

                timestamp = frame_number / fps

                frame_ids = []

                # ==================================================
                # 5. MENGAMBIL INFORMASI SETIAP OBJEK
                # ==================================================

                if (
                    boxes is not None
                    and detection_count > 0
                ):

                    coordinates = (
                        boxes.xyxy
                        .cpu()
                        .tolist()
                    )

                    confidences = (
                        boxes.conf
                        .cpu()
                        .tolist()
                    )

                    class_ids = (
                        boxes.cls
                        .int()
                        .cpu()
                        .tolist()
                    )

                    # Mengambil Track ID
                    if boxes.id is not None:

                        track_ids = (
                            boxes.id
                            .int()
                            .cpu()
                            .tolist()
                        )

                    else:

                        track_ids = (
                            [None] * detection_count
                        )

                    # --------------------------------------------------
                    # Mencatat masing-masing objek
                    # --------------------------------------------------

                    for (
                        coordinate,
                        confidence,
                        class_id,
                        track_id
                    ) in zip(
                        coordinates,
                        confidences,
                        class_ids,
                        track_ids
                    ):

                        x1, y1, x2, y2 = map(
                            float,
                            coordinate
                        )

                        class_id = int(class_id)

                        # Nama kelas
                        if isinstance(result.names, dict):

                            class_name = result.names.get(
                                class_id,
                                str(class_id)
                            )

                        else:

                            class_name = (
                                result.names[class_id]
                            )

                        # Mengelola Track ID
                        if track_id is not None:

                            track_id = int(track_id)

                            unique_track_ids.add(
                                track_id
                            )

                            frame_ids.append(
                                track_id
                            )

                        # --------------------------------------------------
                        # Menulis data objek ke CSV
                        # --------------------------------------------------

                        detection_writer.writerow([

                            frame_number,

                            round(timestamp, 3),

                            (
                                ""
                                if track_id is None
                                else track_id
                            ),

                            class_id,

                            class_name,

                            round(
                                float(confidence),
                                4
                            ),

                            round(x1, 2),
                            round(y1, 2),
                            round(x2, 2),
                            round(y2, 2),

                            round(
                                x2 - x1,
                                2
                            ),

                            round(
                                y2 - y1,
                                2
                            )

                        ])

                    if frame_ids:

                        frames_with_tracking += 1

                # ==================================================
                # 6. MENYIMPAN VIDEO HASIL TRACKING
                # ==================================================

                annotated_frame = result.plot()

                writer.write(
                    annotated_frame
                )

                # ==================================================
                # 7. MENYIMPAN RINGKASAN SETIAP FRAME
                # ==================================================

                frame_writer.writerow([

                    frame_number,

                    round(timestamp, 3),

                    detection_count,

                    ";".join(
                        map(str, frame_ids)
                    )

                ])

                # ==================================================
                # 8. MENAMPILKAN PROGRES
                # ==================================================

                if frame_number % 30 == 0:

                    progress = (
                        frame_number / total_frames * 100
                        if total_frames > 0
                        else 0
                    )

                    print(

                        f"Frame: {frame_number}/{total_frames} "
                        f"({progress:.1f}%) | "
                        f"Deteksi: {detection_count} | "
                        f"Track ID: {len(unique_track_ids)}",

                        flush=True

                    )

    finally:

        cap.release()
        writer.release()

    # ==================================================
    # 9. RINGKASAN HASIL TRACKING
    # ==================================================

    elapsed = time.time() - start_time

    print("\n" + "=" * 55)
    print("TRACKING SELESAI")
    print("=" * 55)

    print(
        "Total frame           :",
        frame_number
    )

    print(
        "Total deteksi frame   :",
        total_detections
    )

    print(
        "Frame dengan tracking :",
        frames_with_tracking
    )

    print(
        "Jumlah Track ID       :",
        len(unique_track_ids)
    )

    print(
        "Waktu pemrosesan      :",
        round(elapsed, 2),
        "detik"
    )

    print("\nVideo hasil:")
    print(output_video)

    print("\nCSV detail setiap objek:")
    print(output_csv)

    print("\nCSV ringkasan setiap frame:")
    print(output_frame_csv)

    print(
        "\nCatatan: jumlah Track ID merupakan jumlah "
        "identitas tracking yang pernah muncul, "
        "bukan jaminan jumlah buah unik."
    )


# ==================================================
# 10. PROGRAM UTAMA
# ==================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Tracking mangga menggunakan "
            "YOLO11n dan ByteTrack"
        )
    )

    parser.add_argument(
        "--source",
        type=str,
        required=True,
        help="Lokasi file video"
    )

    args = parser.parse_args()

    try:

        track_video(args.source)

    except Exception as error:

        print("\nERROR:", error)

        raise SystemExit(1)


if __name__ == "__main__":
    main()