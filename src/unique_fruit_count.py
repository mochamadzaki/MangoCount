from pathlib import Path
import argparse
import csv
from collections import Counter


def read_csv(path):
    with open(path, newline="", encoding="utf-8-sig") as file:
        return list(csv.DictReader(file))


def main():

    parser = argparse.ArgumentParser(
        description="Penghitungan sementara buah mangga unik"
    )

    parser.add_argument("--validation", required=True)
    parser.add_argument("--detections", required=True)

    args = parser.parse_args()

    validation_path = Path(args.validation)
    detections_path = Path(args.detections)

    if not validation_path.exists():
        raise FileNotFoundError(validation_path)

    if not detections_path.exists():
        raise FileNotFoundError(detections_path)

    validation = read_csv(validation_path)
    detections = read_csv(detections_path)

    # Jumlah deteksi untuk setiap Track ID
    detection_counts = Counter(
        row["track_id"].strip()
        for row in detections
        if row["track_id"].strip()
    )

    # Kelompok validasi
    valid = []
    false_positive = []
    review = []
    merge_candidates = []

    known_ids = {
        row["track_id"].strip()
        for row in validation
    }

    for row in validation:

        track_id = row["track_id"].strip()
        status = row["status"].strip().upper()
        canonical_id = row["canonical_id"].strip()

        if track_id not in detection_counts:
            raise ValueError(
                f"Track ID {track_id} tidak ditemukan "
                "dalam CSV deteksi."
            )

        if status == "VALID":

            valid.append({
                "track_id": track_id,
                "canonical_id": canonical_id or track_id,
                "detections": detection_counts[track_id]
            })

        elif status == "FALSE_POSITIVE":

            false_positive.append(track_id)

        elif status == "REVIEW":

            review.append(track_id)

        elif status == "MERGE_CANDIDATE":

            if not canonical_id:
                raise ValueError(
                    f"ID {track_id} belum memiliki canonical_id."
                )

            if canonical_id not in known_ids:
                raise ValueError(
                    f"Canonical ID {canonical_id} "
                    "tidak ditemukan dalam validasi."
                )

            merge_candidates.append({
                "track_id": track_id,
                "canonical_id": canonical_id,
                "detections": detection_counts[track_id]
            })

        else:

            raise ValueError(
                f"Status tidak dikenal: {status}"
            )

    # Identitas valid berdasarkan canonical ID
    confirmed_groups = {}

    for item in valid:

        canonical_id = item["canonical_id"]

        confirmed_groups.setdefault(
            canonical_id, []
        ).append(item["track_id"])

    confirmed_count = len(confirmed_groups)

    # Kandidat yang belum dipastikan sama atau berbeda
    possible_additional = len(merge_candidates)

    # Track yang belum berhasil diverifikasi
    uncertain_count = len(review)

    # Batas sementara, bukan angka final ground truth
    minimum_count = confirmed_count

    maximum_count = (
        confirmed_count
        + possible_additional
        + uncertain_count
    )

    # ==================================================
    # MENYIMPAN LAPORAN IDENTITAS
    # ==================================================

    output_dir = validation_path.parent

    report_path = (
        output_dir / "unique_fruit_report.csv"
    )

    report = []

    for canonical_id, track_ids in confirmed_groups.items():

        report.append({
            "canonical_id": canonical_id,
            "track_ids": ";".join(track_ids),
            "status": "CONFIRMED_VALID"
        })

    for item in merge_candidates:

        report.append({
            "canonical_id": item["canonical_id"],
            "track_ids": item["track_id"],
            "status": "MERGE_CANDIDATE_NOT_CONFIRMED"
        })

    for track_id in review:

        report.append({
            "canonical_id": "",
            "track_ids": track_id,
            "status": "NEEDS_REVIEW"
        })

    for track_id in false_positive:

        report.append({
            "canonical_id": "",
            "track_ids": track_id,
            "status": "EXCLUDED_FALSE_POSITIVE"
        })

    with open(
        report_path,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=[
                "canonical_id",
                "track_ids",
                "status"
            ]
        )

        writer.writeheader()
        writer.writerows(report)

    # ==================================================
    # MENYIMPAN RINGKASAN
    # ==================================================

    summary_path = (
        output_dir / "unique_fruit_summary.csv"
    )

    summary = [
        ("total_track_ids", len(validation)),
        ("confirmed_valid_identities", confirmed_count),
        ("false_positive_ids", len(false_positive)),
        ("unverified_ids", uncertain_count),
        ("merge_candidates", possible_additional),
        ("provisional_minimum_count", minimum_count),
        ("provisional_maximum_count", maximum_count),
        ("final_count_status", "NOT_FINAL")
    ]

    with open(
        summary_path,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.writer(file)

        writer.writerow(["metric", "value"])
        writer.writerows(summary)

    # ==================================================
    # MENAMPILKAN HASIL
    # ==================================================

    print("=" * 60)
    print("UNIQUE FRUIT COUNTING")
    print("=" * 60)

    print("Total Track ID              :", len(validation))
    print("Valid identities            :", confirmed_count)
    print("False positive              :", len(false_positive))
    print("Unverified                  :", uncertain_count)
    print("Merge candidates            :", possible_additional)

    print("\nDAFTAR IDENTITAS VALID")

    for canonical_id, track_ids in confirmed_groups.items():

        print(
            f"MANGO_{canonical_id}: "
            f"Track ID {', '.join(track_ids)}"
        )

    print("\nDAFTAR FALSE POSITIVE:")
    print(false_positive)

    print("\nDAFTAR BELUM TERVERIFIKASI:")
    print(review)

    print("\nKANDIDAT PENGGABUNGAN:")

    for item in merge_candidates:

        print(
            f"ID {item['track_id']} -> "
            f"MANGO_{item['canonical_id']} "
            "(belum dikonfirmasi)"
        )

    print("\n" + "=" * 60)
    print("HASIL PENGHITUNGAN SEMENTARA")
    print("=" * 60)

    print("Jumlah minimum sementara:", minimum_count)
    print("Jumlah maksimum sementara:", maximum_count)

    print("\nStatus: BELUM FINAL")

    print("\nFile hasil:")
    print(report_path)
    print(summary_path)


if __name__ == "__main__":
    main()