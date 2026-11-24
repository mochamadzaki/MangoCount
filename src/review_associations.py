from pathlib import Path
import argparse
import csv
import math
import cv2
import numpy as np


# ==================================================
# 1. MEMBACA DATA DETEKSI
# ==================================================

def load_detections(csv_path):

    detections = {}

    with open(
        csv_path,
        newline="",
        encoding="utf-8"
    ) as file:

        reader = csv.DictReader(file)

        for row in reader:

            track_id = row["track_id"].strip()

            if not track_id:
                continue

            key = (
                int(track_id),
                int(row["frame"])
            )

            detections[key] = row

    return detections


# ==================================================
# 2. MENGAMBIL FRAME VIDEO
# ==================================================

def read_video_frame(cap, frame_number):

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
# 3. MEMBUAT CROP BOUNDING BOX
# ==================================================

def create_crop(frame, detection):

    h, w = frame.shape[:2]

    x1 = float(detection["x1"])
    y1 = float(detection["y1"])
    x2 = float(detection["x2"])
    y2 = float(detection["y2"])

    box_w = max(x2 - x1, 1)
    box_h = max(y2 - y1, 1)

    # Margin agar objek dan lingkungan sekitar terlihat
    margin_x = max(int(box_w * 0.8), 30)
    margin_y = max(int(box_h * 0.8), 30)

    # Membatasi koordinat agar tidak keluar gambar
    left = max(0, int(x1) - margin_x)
    top = max(0, int(y1) - margin_y)

    right = min(w, int(x2) + margin_x)
    bottom = min(h, int(y2) + margin_y)

    crop = frame[top:bottom, left:right].copy()

    if crop.size == 0:

        raise ValueError(
            "Crop kosong. Periksa koordinat bounding box."
        )

    # Koordinat lokal bounding box
    local_x1 = max(0, int(x1) - left)
    local_y1 = max(0, int(y1) - top)

    local_x2 = min(right - left, int(x2) - left)
    local_y2 = min(bottom - top, int(y2) - top)

    # Menggambar bounding box
    cv2.rectangle(
        crop,
        (local_x1, local_y1),
        (local_x2, local_y2),
        (0, 255, 0),
        3
    )

    return crop


# ==================================================
# 4. MENYESUAIKAN UKURAN GAMBAR
# ==================================================

def fit_image(image, width, height):

    if image is None or image.size == 0:

        raise ValueError(
            "Gambar kosong saat proses resize."
        )

    ih, iw = image.shape[:2]

    scale = min(
        width / iw,
        height / ih
    )

    new_w = max(1, int(iw * scale))
    new_h = max(1, int(ih * scale))

    resized = cv2.resize(
        image,
        (new_w, new_h)
    )

    canvas = np.zeros(
        (height, width, 3),
        dtype=np.uint8
    )

    x = (width - new_w) // 2
    y = (height - new_h) // 2

    canvas[
        y:y + new_h,
        x:x + new_w
    ] = resized

    return canvas


# ==================================================
# 5. MEMBUAT GAMBAR PERBANDINGAN
# ==================================================

def create_candidate_review(
    index,
    candidate,
    detections,
    cap,
    frame_cache
):

    id_a = int(candidate["earlier_track_id"])
    id_b = int(candidate["later_track_id"])

    frame_a = int(candidate["earlier_last_frame"])
    frame_b = int(candidate["later_first_frame"])

    # Mengambil data deteksi
    det_a = detections.get((id_a, frame_a))
    det_b = detections.get((id_b, frame_b))

    if det_a is None or det_b is None:

        print(
            f"Kandidat {index}: "
            "data deteksi tidak ditemukan."
        )

        return None

    # --------------------------------------------------
    # Mengambil frame pertama
    # --------------------------------------------------

    if frame_a not in frame_cache:

        frame_cache[frame_a] = read_video_frame(
            cap,
            frame_a
        )

    # --------------------------------------------------
    # Mengambil frame kedua
    # --------------------------------------------------

    if frame_b not in frame_cache:

        frame_cache[frame_b] = read_video_frame(
            cap,
            frame_b
        )

    # --------------------------------------------------
    # Membuat crop
    # --------------------------------------------------

    crop_a = create_crop(
        frame_cache[frame_a],
        det_a
    )

    crop_b = create_crop(
        frame_cache[frame_b],
        det_b
    )

    # --------------------------------------------------
    # Ukuran panel
    # --------------------------------------------------

    panel_w = 350
    panel_h = 220

    # CANVAS DIPERBAIKI MENJADI 305 PIXEL
    canvas = np.zeros(
        (305, 730, 3),
        dtype=np.uint8
    )

    # --------------------------------------------------
    # Informasi kandidat
    # --------------------------------------------------

    distance = candidate["normalized_distance"]
    gap = candidate["gap_frames"]

    title = (
        f"Candidate {index}: "
        f"ID {id_a} -> ID {id_b}"
    )

    cv2.putText(
        canvas,
        title,
        (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    cv2.putText(
        canvas,
        f"Gap: {gap} | Distance: {distance}",
        (10, 48),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 255, 255),
        1
    )

    # Informasi frame pertama
    cv2.putText(
        canvas,
        f"ID {id_a} - Frame {frame_a}",
        (10, 75),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 255, 0),
        1
    )

    # Informasi frame kedua
    cv2.putText(
        canvas,
        f"ID {id_b} - Frame {frame_b}",
        (375, 75),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 255, 0),
        1
    )

    # --------------------------------------------------
    # Resize crop
    # --------------------------------------------------

    panel_a = fit_image(
        crop_a,
        panel_w,
        panel_h
    )

    panel_b = fit_image(
        crop_b,
        panel_w,
        panel_h
    )

    # --------------------------------------------------
    # Menempelkan gambar ke canvas
    # --------------------------------------------------

    canvas[
        85:305,
        10:360
    ] = panel_a

    canvas[
        85:305,
        375:725
    ] = panel_b

    return canvas


# ==================================================
# 6. PROGRAM UTAMA
# ==================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Visualisasi kandidat Tracklet Association"
        )
    )

    parser.add_argument(
        "--video",
        required=True,
        help="Lokasi video asli"
    )

    parser.add_argument(
        "--detections",
        required=True,
        help="Lokasi CSV detail deteksi"
    )

    parser.add_argument(
        "--candidates",
        required=True,
        help="Lokasi CSV kandidat asosiasi"
    )

    args = parser.parse_args()

    video_path = Path(args.video)
    detections_path = Path(args.detections)
    candidates_path = Path(args.candidates)

    # --------------------------------------------------
    # Pemeriksaan file
    # --------------------------------------------------

    for path in [
        video_path,
        detections_path,
        candidates_path
    ]:

        if not path.exists():

            raise FileNotFoundError(
                f"File tidak ditemukan: {path}"
            )

    # --------------------------------------------------
    # Folder output
    # --------------------------------------------------

    output_dir = (
        candidates_path.parent
        / "association_review"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    print("=" * 60)
    print("TRACKLET ASSOCIATION VISUAL REVIEW")
    print("=" * 60)

    # --------------------------------------------------
    # Membaca data deteksi
    # --------------------------------------------------

    print("\nMembaca data deteksi...")

    detections = load_detections(
        detections_path
    )

    print(
        "Jumlah data deteksi:",
        len(detections)
    )

    # --------------------------------------------------
    # Membaca kandidat asosiasi
    # --------------------------------------------------

    with open(
        candidates_path,
        newline="",
        encoding="utf-8"
    ) as file:

        candidates = list(
            csv.DictReader(file)
        )

    print(
        "Jumlah kandidat:",
        len(candidates)
    )

    if not candidates:

        print(
            "\nTidak ada kandidat untuk diperiksa."
        )

        return

    # --------------------------------------------------
    # Membuka video
    # --------------------------------------------------

    cap = cv2.VideoCapture(
        str(video_path)
    )

    if not cap.isOpened():

        raise RuntimeError(
            "Video gagal dibuka."
        )

    frame_cache = {}

    review_images = []

    try:

        # ==================================================
        # MEMPROSES SETIAP KANDIDAT
        # ==================================================

        for index, candidate in enumerate(
            candidates,
            start=1
        ):

            print(
                f"\nMemproses kandidat {index}/"
                f"{len(candidates)}..."
            )

            try:

                review = create_candidate_review(
                    index,
                    candidate,
                    detections,
                    cap,
                    frame_cache
                )

                if review is None:
                    continue

                # ------------------------------------------
                # Simpan gambar individual
                # ------------------------------------------

                output_file = (
                    output_dir
                    / f"candidate_{index:02d}.jpg"
                )

                cv2.imwrite(
                    str(output_file),
                    review
                )

                review_images.append(review)

                print(
                    f"Berhasil: ID "
                    f"{candidate['earlier_track_id']} -> "
                    f"ID {candidate['later_track_id']}"
                )

            except Exception as error:

                print(
                    f"Kandidat {index} gagal: {error}"
                )

    finally:

        cap.release()

    # ==================================================
    # 7. MEMBUAT CONTACT SHEET
    # ==================================================

    if not review_images:

        print(
            "\nTidak ada gambar kandidat yang berhasil dibuat."
        )

        return

    columns = 2

    rows = math.ceil(
        len(review_images) / columns
    )

    # Ukuran canvas diperbaiki
    panel_height = 305
    panel_width = 730

    sheet = np.zeros(
        (
            rows * panel_height,
            columns * panel_width,
            3
        ),
        dtype=np.uint8
    )

    # --------------------------------------------------
    # Menempelkan setiap kandidat
    # --------------------------------------------------

    for i, image in enumerate(review_images):

        x = (
            i % columns
        ) * panel_width

        y = (
            i // columns
        ) * panel_height

        sheet[
            y:y + panel_height,
            x:x + panel_width
        ] = image

    # --------------------------------------------------
    # Menyimpan contact sheet
    # --------------------------------------------------

    contact_sheet = (
        output_dir
        / "association_contact_sheet.jpg"
    )

    cv2.imwrite(
        str(contact_sheet),
        sheet
    )

    # ==================================================
    # 8. RINGKASAN
    # ==================================================

    print("\n" + "=" * 60)
    print("VISUAL REVIEW SELESAI")
    print("=" * 60)

    print(
        "Jumlah kandidat berhasil:",
        len(review_images)
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
        "\nCatatan: kandidat asosiasi belum otomatis "
        "digabung. Pemeriksaan visual tetap diperlukan."
    )


# ==================================================
# 9. PROGRAM UTAMA
# ==================================================

if __name__ == "__main__":
    main()