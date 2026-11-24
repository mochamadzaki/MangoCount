from pathlib import Path
from collections import defaultdict
import argparse
import csv
import math
import cv2
import numpy as np


# ==================================================
# 1. KONFIGURASI
# ==================================================

TILE_WIDTH = 420
TILE_HEIGHT = 300

CONTACT_COLUMNS = 3

CROP_PADDING = 0.6

OUTPUT_FOLDER = "track_review"


# ==================================================
# 2. MEMBACA DATA CSV
# ==================================================

def load_tracks(csv_path):

    tracks = defaultdict(list)

    required_columns = {
        "frame",
        "track_id",
        "confidence",
        "x1",
        "y1",
        "x2",
        "y2",
        "width",
        "height"
    }

    with open(
        csv_path,
        newline="",
        encoding="utf-8"
    ) as file:

        reader = csv.DictReader(file)

        if not required_columns.issubset(
            set(reader.fieldnames or [])
        ):
            raise ValueError(
                "Kolom CSV tidak lengkap."
            )

        for row in reader:

            track_id = row["track_id"].strip()

            if not track_id:
                continue

            detection = {

                "frame": int(row["frame"]),

                "track_id": int(track_id),

                "confidence": float(
                    row["confidence"]
                ),

                "x1": float(row["x1"]),
                "y1": float(row["y1"]),
                "x2": float(row["x2"]),
                "y2": float(row["y2"]),

                "width": float(row["width"]),
                "height": float(row["height"])

            }

            tracks[int(track_id)].append(
                detection
            )

    # Urutkan berdasarkan frame
    for track_id in tracks:

        tracks[track_id].sort(
            key=lambda item: item["frame"]
        )

    return tracks


# ==================================================
# 3. MEMBACA FRAME VIDEO
# ==================================================

def read_frame(cap, frame_number):

    cap.set(
        cv2.CAP_PROP_POS_FRAMES,
        frame_number - 1
    )

    success, frame = cap.read()

    if not success:

        raise RuntimeError(
            f"Gagal membaca frame {frame_number}"
        )

    return frame


# ==================================================
# 4. MEMBUAT CROP BOUNDING BOX
# ==================================================

def crop_detection(frame, detection):

    height, width = frame.shape[:2]

    x1 = detection["x1"]
    y1 = detection["y1"]

    x2 = detection["x2"]
    y2 = detection["y2"]

    box_width = max(x2 - x1, 1)
    box_height = max(y2 - y1, 1)

    # Margin tambahan agar konteks sekitar terlihat
    margin_x = int(
        box_width * CROP_PADDING
    )

    margin_y = int(
        box_height * CROP_PADDING
    )

    left = max(
        0,
        int(x1 - margin_x)
    )

    top = max(
        0,
        int(y1 - margin_y)
    )

    right = min(
        width,
        int(x2 + margin_x)
    )

    bottom = min(
        height,
        int(y2 + margin_y)
    )

    crop = frame[
        top:bottom,
        left:right
    ].copy()

    if crop.size == 0:

        raise ValueError(
            "Bounding box menghasilkan crop kosong."
        )

    # Koordinat bounding box pada crop
    local_x1 = int(x1 - left)
    local_y1 = int(y1 - top)

    local_x2 = int(x2 - left)
    local_y2 = int(y2 - top)

    # Gambar bounding box
    cv2.rectangle(
        crop,
        (local_x1, local_y1),
        (local_x2, local_y2),
        (0, 255, 0),
        3
    )

    return crop


# ==================================================
# 5. RESIZE GAMBAR
# ==================================================

def resize_image(image, target_width, target_height):

    h, w = image.shape[:2]

    scale = min(
        target_width / w,
        target_height / h
    )

    new_width = max(
        1,
        int(w * scale)
    )

    new_height = max(
        1,
        int(h * scale)
    )

    resized = cv2.resize(
        image,
        (new_width, new_height)
    )

    canvas = np.zeros(
        (target_height, target_width, 3),
        dtype=np.uint8
    )

    x = (target_width - new_width) // 2
    y = (target_height - new_height) // 2

    canvas[
        y:y + new_height,
        x:x + new_width
    ] = resized

    return canvas


# ==================================================
# 6. MEMBUAT PANEL GAMBAR
# ==================================================

def create_tile(title, detection, crop):

    tile = np.zeros(
        (TILE_HEIGHT, TILE_WIDTH, 3),
        dtype=np.uint8
    )

    cv2.putText(
        tile,
        title,
        (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )

    information = (
        f"Frame: {detection['frame']} | "
        f"Confidence: {detection['confidence']:.3f}"
    )

    cv2.putText(
        tile,
        information,
        (10, 48),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (0, 255, 255),
        1,
        cv2.LINE_AA
    )

    resized = resize_image(
        crop,
        TILE_WIDTH - 20,
        TILE_HEIGHT - 65
    )

    tile[
        55:TILE_HEIGHT - 10,
        10:TILE_WIDTH - 10
    ] = resized

    return tile


# ==================================================
# 7. MEMILIH FRAME SETIAP TRACK ID
# ==================================================

def select_track_frames(detections):

    first_detection = detections[0]

    best_detection = max(
        detections,
        key=lambda item: item["confidence"]
    )

    last_detection = detections[-1]

    return [

        ("FIRST", first_detection),

        ("BEST CONFIDENCE", best_detection),

        ("LAST", last_detection)

    ]


# ==================================================
# 8. MEMBUAT REVIEW SETIAP TRACK ID
# ==================================================

def review_track(
    cap,
    track_id,
    detections,
    output_dir
):

    selected_frames = select_track_frames(
        detections
    )

    tiles = []

    for label, detection in selected_frames:

        frame = read_frame(
            cap,
            detection["frame"]
        )

        crop = crop_detection(
            frame,
            detection
        )

        title = (
            f"ID {track_id} - {label}"
        )

        tile = create_tile(
            title,
            detection,
            crop
        )

        tiles.append(tile)

    # Gabungkan FIRST, BEST, LAST
    panel = cv2.hconcat(tiles)

    output_file = (
        output_dir / f"track_{track_id:03d}.jpg"
    )

    cv2.imwrite(
        str(output_file),
        panel
    )

    return tiles[1], selected_frames


# ==================================================
# 9. MENJALANKAN REVIEW SELURUH TRACK
# ==================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Visual review seluruh Track ID"
        )
    )

    parser.add_argument(
        "--video",
        required=True
    )

    parser.add_argument(
        "--csv",
        required=True
    )

    args = parser.parse_args()

    video_path = Path(args.video)
    csv_path = Path(args.csv)

    if not video_path.exists():
        raise FileNotFoundError(video_path)

    if not csv_path.exists():
        raise FileNotFoundError(csv_path)

    output_dir = (
        csv_path.parent / OUTPUT_FOLDER
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    print("=" * 60)
    print("TRACK ID VISUAL REVIEW")
    print("=" * 60)

    # --------------------------------------------------
    # Baca data
    # --------------------------------------------------

    tracks = load_tracks(csv_path)

    print(
        "Jumlah Track ID:",
        len(tracks)
    )

    # --------------------------------------------------
    # Buka video
    # --------------------------------------------------

    cap = cv2.VideoCapture(
        str(video_path)
    )

    if not cap.isOpened():

        raise RuntimeError(
            "Video gagal dibuka."
        )

    best_tiles = []
    summary = []

    try:

        # --------------------------------------------------
        # Proses setiap Track ID
        # --------------------------------------------------

        for index, (track_id, detections) in enumerate(
            sorted(tracks.items()),
            start=1
        ):

            print(
                f"Memproses ID {track_id} "
                f"({index}/{len(tracks)})..."
            )

            best_tile, selected = review_track(
                cap,
                track_id,
                detections,
                output_dir
            )

            best_tiles.append(best_tile)

            frames = [
                item["frame"]
                for item in detections
            ]

            confidences = [
                item["confidence"]
                for item in detections
            ]

            best_detection = max(
                detections,
                key=lambda item: item["confidence"]
            )

            summary.append({

                "track_id": track_id,

                "first_frame": min(frames),

                "last_frame": max(frames),

                "detection_count": len(detections),

                "mean_confidence": round(
                    sum(confidences) / len(confidences),
                    4
                ),

                "max_confidence": round(
                    max(confidences),
                    4
                ),

                "best_frame": best_detection["frame"]

            })

    finally:

        cap.release()

    # ==================================================
    # 10. CONTACT SHEET SELURUH TRACK ID
    # ==================================================

    columns = CONTACT_COLUMNS

    rows = math.ceil(
        len(best_tiles) / columns
    )

    sheet = np.zeros(
        (
            rows * TILE_HEIGHT,
            columns * TILE_WIDTH,
            3
        ),
        dtype=np.uint8
    )

    for index, tile in enumerate(best_tiles):

        x = (
            index % columns
        ) * TILE_WIDTH

        y = (
            index // columns
        ) * TILE_HEIGHT

        sheet[
            y:y + TILE_HEIGHT,
            x:x + TILE_WIDTH
        ] = tile

    contact_sheet = (
        output_dir / "all_tracks_contact_sheet.jpg"
    )

    cv2.imwrite(
        str(contact_sheet),
        sheet
    )

    # ==================================================
    # 11. SIMPAN RINGKASAN TRACK
    # ==================================================

    summary_file = (
        output_dir / "all_tracks_summary.csv"
    )

    with open(
        summary_file,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=summary[0].keys()
        )

        writer.writeheader()
        writer.writerows(summary)

    # ==================================================
    # 12. RINGKASAN
    # ==================================================

    print("\n" + "=" * 60)
    print("VISUAL REVIEW SELESAI")
    print("=" * 60)

    print(
        "Total Track ID:",
        len(tracks)
    )

    print(
        "\nFolder hasil:",
        output_dir
    )

    print(
        "\nContact sheet:",
        contact_sheet
    )

    print(
        "\nRingkasan CSV:",
        summary_file
    )


if __name__ == "__main__":
    main()