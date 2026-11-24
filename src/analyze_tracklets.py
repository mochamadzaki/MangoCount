from pathlib import Path
from collections import defaultdict
import argparse
import csv
import statistics


def analyze_tracklets(csv_path):

    csv_path = Path(csv_path)

    if not csv_path.exists():
        raise FileNotFoundError(csv_path)

    tracklets = defaultdict(list)
    untracked_count = 0

    # Membaca CSV hasil tracking
    with open(csv_path, newline="", encoding="utf-8") as file:

        reader = csv.DictReader(file)

        for row in reader:

            track_id = row["track_id"].strip()

            if not track_id:
                untracked_count += 1
                continue

            tracklets[track_id].append({

                "frame": int(row["frame"]),
                "timestamp": float(row["timestamp_seconds"]),
                "confidence": float(row["confidence"]),

                "x1": float(row["x1"]),
                "y1": float(row["y1"]),
                "x2": float(row["x2"]),
                "y2": float(row["y2"]),

                "width": float(row["width"]),
                "height": float(row["height"])

            })

    if not tracklets:
        print("Tidak ditemukan Track ID.")
        return

    summary = []

    for track_id, detections in tracklets.items():

        frames = sorted(set(
            item["frame"] for item in detections
        ))

        first_frame = min(frames)
        last_frame = max(frames)

        span = last_frame - first_frame + 1

        confidence_values = [
            item["confidence"] for item in detections
        ]

        center_x = [
            (item["x1"] + item["x2"]) / 2
            for item in detections
        ]

        center_y = [
            (item["y1"] + item["y2"]) / 2
            for item in detections
        ]

        avg_width = statistics.mean(
            item["width"] for item in detections
        )

        avg_height = statistics.mean(
            item["height"] for item in detections
        )

        coverage = len(frames) / span * 100

        summary.append({

            "track_id": track_id,
            "first_frame": first_frame,
            "last_frame": last_frame,
            "duration_frames": span,
            "detected_frames": len(frames),
            "detection_rows": len(detections),
            "coverage_percent": round(coverage, 2),

            "mean_confidence": round(
                statistics.mean(confidence_values), 4
            ),

            "min_confidence": round(
                min(confidence_values), 4
            ),

            "max_confidence": round(
                max(confidence_values), 4
            ),

            "avg_center_x": round(
                statistics.mean(center_x), 2
            ),

            "avg_center_y": round(
                statistics.mean(center_y), 2
            ),

            "avg_width": round(avg_width, 2),
            "avg_height": round(avg_height, 2)

        })

    summary.sort(
        key=lambda item: item["track_id"]
    )

    # Menyimpan hasil analisis
    output_path = (
        csv_path.parent / "tracklet_summary.csv"
    )

    with open(
        output_path,
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

    # Menampilkan ringkasan
    print("=" * 60)
    print("TRACKLET ANALYSIS")
    print("=" * 60)

    print("Jumlah Track ID       :", len(summary))
    print("Total tracklet        :", sum(
        item["detection_rows"] for item in summary
    ))

    print("Deteksi tanpa Track ID:", untracked_count)

    print("\nRINCIAN SETIAP TRACKLET")
    print("-" * 60)

    for item in summary:

        print(
            f"ID {item['track_id']:>4} | "
            f"Frame {item['first_frame']:>4}-"
            f"{item['last_frame']:<4} | "
            f"Deteksi {item['detection_rows']:>4} | "
            f"Confidence {item['mean_confidence']:.3f} | "
            f"Coverage {item['coverage_percent']:.1f}%"
        )

    print("\nHasil tersimpan:")
    print(output_path)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--csv",
        required=True,
        help="Lokasi CSV detail hasil tracking"
    )

    args = parser.parse_args()

    analyze_tracklets(args.csv)