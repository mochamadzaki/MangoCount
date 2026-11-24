from pathlib import Path
from collections import defaultdict
import argparse
import csv
import math
import statistics


# ==================================================
# KONFIGURASI
# ==================================================

MAX_GAP_FRAMES = 90
MAX_NORMALIZED_DISTANCE = 2.5
MAX_SIZE_RATIO = 4.0
SEGMENT_GAP = 5


# ==================================================
# MEMBACA DATA
# ==================================================

def load_detections(csv_path):

    tracks = defaultdict(list)

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

            tracks[int(track_id)].append({
                "frame": int(row["frame"]),
                "confidence": float(row["confidence"]),
                "x1": float(row["x1"]),
                "y1": float(row["y1"]),
                "x2": float(row["x2"]),
                "y2": float(row["y2"]),
                "width": float(row["width"]),
                "height": float(row["height"])
            })

    for track_id in tracks:
        tracks[track_id].sort(
            key=lambda item: item["frame"]
        )

    return tracks


# ==================================================
# MEMECAH TRACK MENJADI SEGMEN
# ==================================================

def build_segments(tracks):

    segments = []

    for track_id, detections in tracks.items():

        current = []

        for detection in detections:

            if current:

                gap = (
                    detection["frame"]
                    - current[-1]["frame"]
                )

                if gap > SEGMENT_GAP:

                    segments.append({
                        "track_id": track_id,
                        "detections": current
                    })

                    current = []

            current.append(detection)

        if current:

            segments.append({
                "track_id": track_id,
                "detections": current
            })

    return segments


# ==================================================
# INFORMASI SEGMEN
# ==================================================

def segment_information(segment):

    detections = segment["detections"]

    first = detections[0]
    last = detections[-1]

    def center(item):

        return (
            (item["x1"] + item["x2"]) / 2,
            (item["y1"] + item["y2"]) / 2
        )

    first_center = center(first)
    last_center = center(last)

    areas = [
        item["width"] * item["height"]
        for item in detections
    ]

    return {
        "track_id": segment["track_id"],
        "first_frame": first["frame"],
        "last_frame": last["frame"],
        "detections": len(detections),

        "first_x": first_center[0],
        "first_y": first_center[1],

        "last_x": last_center[0],
        "last_y": last_center[1],

        "first_width": first["width"],
        "first_height": first["height"],

        "last_width": last["width"],
        "last_height": last["height"],

        "mean_area": statistics.mean(areas),

        "mean_confidence": statistics.mean(
            item["confidence"]
            for item in detections
        )
    }


# ==================================================
# MENCARI KANDIDAT ASOSIASI
# ==================================================

def find_candidates(segments):

    information = [
        segment_information(segment)
        for segment in segments
    ]

    candidates = []

    for i, a in enumerate(information):

        for b in information[i + 1:]:

            # Tidak membandingkan ID yang sama
            if a["track_id"] == b["track_id"]:
                continue

            # Mengurutkan berdasarkan waktu
            if a["last_frame"] < b["first_frame"]:
                earlier, later = a, b

            elif b["last_frame"] < a["first_frame"]:
                earlier, later = b, a

            else:
                # Rentang waktu tumpang tindih.
                # Tidak otomatis dianggap objek sama.
                continue

            gap = (
                later["first_frame"]
                - earlier["last_frame"]
                - 1
            )

            if gap > MAX_GAP_FRAMES:
                continue

            # Jarak posisi akhir dan awal
            distance = math.hypot(
                later["first_x"] - earlier["last_x"],
                later["first_y"] - earlier["last_y"]
            )

            # Normalisasi terhadap ukuran buah
            avg_diagonal = (
                math.hypot(
                    earlier["last_width"],
                    earlier["last_height"]
                )
                +
                math.hypot(
                    later["first_width"],
                    later["first_height"]
                )
            ) / 2

            if avg_diagonal <= 0:
                continue

            normalized_distance = (
                distance / avg_diagonal
            )

            size_ratio = max(
                earlier["mean_area"],
                later["mean_area"]
            ) / max(
                min(
                    earlier["mean_area"],
                    later["mean_area"]
                ),
                1
            )

            if (
                normalized_distance
                <= MAX_NORMALIZED_DISTANCE
                and size_ratio <= MAX_SIZE_RATIO
            ):

                candidates.append({

                    "earlier_track_id":
                        earlier["track_id"],

                    "later_track_id":
                        later["track_id"],

                    "earlier_last_frame":
                        earlier["last_frame"],

                    "later_first_frame":
                        later["first_frame"],

                    "gap_frames": gap,

                    "normalized_distance":
                        round(normalized_distance, 3),

                    "size_ratio":
                        round(size_ratio, 3),

                    "earlier_confidence":
                        round(
                            earlier["mean_confidence"],
                            3
                        ),

                    "later_confidence":
                        round(
                            later["mean_confidence"],
                            3
                        ),

                    "status": "NEEDS_VISUAL_REVIEW"

                })

    candidates.sort(
        key=lambda item: (
            item["normalized_distance"],
            item["gap_frames"]
        )
    )

    return information, candidates


# ==================================================
# MENJALANKAN ANALISIS
# ==================================================

def main():

    parser = argparse.ArgumentParser(
        description="Tracklet Association Analysis"
    )

    parser.add_argument(
        "--csv",
        required=True,
        help="CSV detail hasil tracking"
    )

    args = parser.parse_args()

    csv_path = Path(args.csv)

    if not csv_path.exists():
        raise FileNotFoundError(csv_path)

    tracks = load_detections(csv_path)

    segments = build_segments(tracks)

    information, candidates = find_candidates(
        segments
    )

    output_dir = csv_path.parent

    # --------------------------------------------------
    # CSV segmen
    # --------------------------------------------------

    segment_file = (
        output_dir / "tracklet_segments.csv"
    )

    with open(
        segment_file,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        fieldnames = [
            "track_id",
            "first_frame",
            "last_frame",
            "detections",
            "first_x",
            "first_y",
            "last_x",
            "last_y",
            "mean_confidence"
        ]

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames
        )

        writer.writeheader()

        for item in information:

            writer.writerow({
                key: item[key]
                for key in fieldnames
            })

    # --------------------------------------------------
    # CSV kandidat asosiasi
    # --------------------------------------------------

    candidate_file = (
        output_dir / "association_candidates.csv"
    )

    if candidates:

        with open(
            candidate_file,
            "w",
            newline="",
            encoding="utf-8"
        ) as file:

            writer = csv.DictWriter(
                file,
                fieldnames=candidates[0].keys()
            )

            writer.writeheader()
            writer.writerows(candidates)

    else:

        candidate_file.write_text(
            "Tidak ditemukan kandidat asosiasi.\n",
            encoding="utf-8"
        )

    # --------------------------------------------------
    # RINGKASAN
    # --------------------------------------------------

    print("=" * 60)
    print("TRACKLET ASSOCIATION ANALYSIS")
    print("=" * 60)

    print("Jumlah Track ID       :", len(tracks))
    print("Jumlah segmen         :", len(segments))
    print("Kandidat asosiasi     :", len(candidates))

    print("\nDAFTAR KANDIDAT")

    for item in candidates:

        print(
            f"ID {item['earlier_track_id']} -> "
            f"ID {item['later_track_id']} | "
            f"Gap: {item['gap_frames']} frame | "
            f"Jarak: {item['normalized_distance']}"
        )

    print("\nHasil tersimpan:")
    print(segment_file)
    print(candidate_file)

    print(
        "\nPENTING: Kandidat belum otomatis digabung. "
        "Setiap pasangan membutuhkan pemeriksaan visual."
    )


if __name__ == "__main__":
    main()