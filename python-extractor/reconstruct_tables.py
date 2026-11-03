
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from collections import Counter
from typing import Any


# ============================================================
# CONFIGURATION
# ============================================================

TARGET_IDS = {51, 52, 53, 54, 89, 131, 148}

# Semua nilai di bawah disusun berdasarkan hasil pemeriksaan
# koordinat PDF yang telah dikirimkan.
#
# Jangan mengubah tanda "-" / "–" / "..." menjadi nilai kosong.
# Jangan mengubah angka berdasarkan asumsi atau perhitungan sendiri.


# ============================================================
# TABLE RECONSTRUCTION SPECIFICATIONS
# ============================================================

def education_headers():
    return [
        "Tingkat Pendidikan",
        "Negeri 2023/2024",
        "Negeri 2024/2025",
        "Swasta 2023/2024",
        "Swasta 2024/2025",
        "Jumlah 2023/2024",
        "Jumlah 2024/2025",
    ]


TABLE_SPECS: dict[int, dict[str, Any]] = {

    # --------------------------------------------------------
    # ID 51 - PAGE 100
    # --------------------------------------------------------
    51: {
        "headers": education_headers(),
        "rows": [
            ["Taman Kanak-Kanak (TK)", "1", "1", "18", "21", "19", "22"],
            ["Raudatul Athfal (RA)", "–", "–", "3", "...", "3", "..."],
            ["Sekolah Dasar (SD)", "69", "69", "3", "4", "72", "73"],
            ["Madrasah Ibtidaiyah (MI)", "1", "...", "–", "...", "1", "..."],
            ["Sekolah Menengah Pertama (SMP)", "15", "15", "8", "8", "23", "23"],
            ["Madrasah Tsanawiyah (MTs)", "1", "...", "1", "...", "2", "..."],
            ["Sekolah Menengah Atas (SMA)", "3", "3", "3", "2", "6", "5"],
            ["Sekolah Menengah Kejuruan (SMK)", "1", "1", "3", "3", "4", "4"],
            ["Madrasah Aliyah (MA)", "1", "...", "–", "...", "1", "..."],
        ],
        "header_count": 1,
        "basis": "Koordinat PDF halaman 100",
    },

    # --------------------------------------------------------
    # ID 52 - PAGE 101
    # --------------------------------------------------------
    52: {
        "headers": education_headers(),
        "rows": [
            ["Taman Kanak-Kanak (TK)", "4", "5", "68", "65", "72", "70"],
            ["Raudatul Athfal (RA)", "–", "–", "24", "...", "24", "..."],
            ["Sekolah Dasar (SD)", "530", "549", "41", "44", "571", "593"],
            ["Madrasah Ibtidaiyah (MI)", "52", "...", "–", "...", "52", "..."],
            ["Sekolah Menengah Pertama (SMP)", "192", "196", "48", "50", "240", "246"],
            ["Madrasah Tsanawiyah (MTs)", "30", "...", "11", "...", "41", "..."],
            ["Sekolah Menengah Atas (SMA)", "84", "90", "35", "34", "119", "124"],
            ["Sekolah Menengah Kejuruan (SMK)", "38", "43", "30", "25", "68", "68"],
            ["Madrasah Aliyah (MA)", "37", "...", "–", "...", "37", "..."],
        ],
        "header_count": 1,
        "basis": "Koordinat PDF halaman 101",
    },

    # --------------------------------------------------------
    # ID 53 - PAGE 102
    # --------------------------------------------------------
    53: {
        "headers": education_headers(),
        "rows": [
            ["Taman Kanak-Kanak (TK)", "28", "31", "875", "840", "903", "871"],
            ["Raudatul Athfal (RA)", "–", "–", "217", "...", "217", "..."],
            ["Sekolah Dasar (SD)", "7.141", "7.121", "888", "627", "8.029", "7.748"],
            ["Madrasah Ibtidaiyah (MI)", "705", "...", "–", "...", "705", "..."],
            ["Sekolah Menengah Pertama (SMP)", "2.964", "3.008", "759", "765", "3.723", "3.773"],
            ["Madrasah Tsanawiyah (MTs)", "377", "...", "45", "...", "422", "..."],
            ["Sekolah Menengah Atas (SMA)", "1.815", "1.921", "619", "614", "2.434", "2.535"],
            ["Sekolah Menengah Kejuruan (SMK)", "837", "869", "630", "697", "1.467", "1.566"],
            ["Madrasah Aliyah (MA)", "357", "...", "–", "...", "357", "..."],
        ],
        "header_count": 1,
        "basis": "Koordinat PDF halaman 102",
    },

    # --------------------------------------------------------
    # ID 54 - PAGE 103
    # --------------------------------------------------------
    54: {
        "headers": [
            "Jenis Sarana Kesehatan",
            "2022",
            "2023",
            "2024",
        ],
        "rows": [
            ["Rumah Sakit", "2", "2", "2"],
            ["Puskesmas Rawat Inap", "1", "1", "1"],
            ["Puskesmas Tanpa Rawat Inap", "2", "2", "2"],
            ["Apotek", "4", "4", "4"],
        ],
        "header_count": 1,
        "basis": "Koordinat PDF halaman 103",
    },

    # --------------------------------------------------------
    # ID 89 - PAGE 139
    # --------------------------------------------------------
    89: {
        "headers": [
            "Kelompok",
            "Jumlah Pelanggan",
            "Banyaknya Air Terjual (m3)",
            "Nilai Penjualan (rupiah)",
        ],
        "rows": [
            ["Rumah Tangga", "5.788", "1.181.635", "8.249.766.150"],
            ["Umum", "–", "–", "–"],
            ["Badan Sosial", "108", "83.912", "491.389.400"],
            ["Pemerintahan", "73", "28.317", "327.159.600"],
            ["Usaha/Industri", "520", "113.497", "1.456.446.200"],
            ["Kecamatan Kapuas", "6.489", "1.407.361", "10.524.761.350"],
        ],
        "header_count": 1,
        "basis": "Koordinat PDF halaman 139",
    },

    # --------------------------------------------------------
    # ID 131 - PAGE 192
    # --------------------------------------------------------
    131: {
        "headers": [
            "Jenis Data",
            "Rincian",
            "Banyaknya",
        ],
        "rows": [
            ["Banyaknya Surat", "", ""],
            ["", "a. Dikirim", "18.730"],
            ["", "b. Diterima", "17.688"],
            ["Banyaknya Wesel", "", ""],
            ["", "a. Dikirim", "4.158"],
            ["", "b. Diterima", "492"],
            ["Paket Pos yang Dikirim", "", ""],
            ["", "a. Jumlah", "14.828"],
        ],
        "header_count": 1,
        "basis": "Koordinat PDF halaman 192",
    },

    # --------------------------------------------------------
    # ID 148 - PAGE 197
    # --------------------------------------------------------
    148: {
        "headers": [
            "Kode Wilayah",
            "Desa/Kelurahan",
            "Kekuatan Sinyal Telepon Seluler",
            "Jenis Sinyal Internet Telepon Seluler",
        ],
        "rows": [
            ["6105060013", "Tanjung Kapuas", "Sinyal kuat", "5G/4G/LTE"],
            ["6105060014", "Tanjung Sekayam", "Sinyal kuat", "5G/4G/LTE"],
            ["6105060015", "Ilir Kota", "Sinyal sangat kuat", "5G/4G/LTE"],
            ["6105060016", "Beringin", "Sinyal kuat", "5G/4G/LTE"],
            ["6105060017", "Bunut", "Sinyal sangat kuat", "3G/H/H+/EVDO"],
            ["6105060020", "Sungai Sengkuang", "Sinyal kuat", "5G/4G/LTE"],
        ],
        "header_count": 1,
        "basis": "Koordinat PDF halaman 197",
    },
}


# ============================================================
# JSONL UTILITIES
# ============================================================

def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []

    if not path.exists():
        raise FileNotFoundError(f"File tidak ditemukan: {path}")

    with path.open("r", encoding="utf-8") as file:
        for line_no, line in enumerate(file, start=1):
            if not line.strip():
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"JSON rusak pada baris {line_no}: {exc}"
                ) from exc

            if not isinstance(record, dict):
                raise ValueError(
                    f"Baris {line_no} bukan objek JSON."
                )

            records.append(record)

    return records


def write_jsonl(
    path: Path,
    records: list[dict[str, Any]]
) -> None:

    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(
                json.dumps(record, ensure_ascii=False) + "\n"
            )


# ============================================================
# RECONSTRUCTION
# ============================================================

def reconstruct_one_table(
    record: dict[str, Any],
    spec: dict[str, Any]
) -> dict[str, Any]:

    result = copy.deepcopy(record)

    original_rows = (
        record.get("clean_rows")
        or record.get("rows")
        or record.get("table")
        or []
    )

    headers = copy.deepcopy(spec["headers"])
    rows = copy.deepcopy(spec["rows"])

    # Validasi jumlah kolom.
    expected_columns = len(headers)

    for row_index, row in enumerate(rows):
        if len(row) != expected_columns:
            raise ValueError(
                f"ID {record.get('table_id')}: "
                f"baris {row_index} memiliki {len(row)} kolom, "
                f"seharusnya {expected_columns}."
            )

    result["original_clean_rows"] = copy.deepcopy(original_rows)
    result["reconstructed_rows"] = [headers] + rows
    result["flattened_header"] = headers
    result["reconstruction_header_count"] = 1

    result["reconstruction_method"] = (
        "MANUAL_COORDINATE_BASED"
    )

    result["reconstruction_basis"] = spec["basis"]

    result["reconstruction_changes"] = [
        {
            "type": "table_reconstruction",
            "description": (
                "Struktur tabel dibentuk kembali berdasarkan "
                "hasil pemeriksaan koordinat teks PDF."
            ),
            "original_row_count": len(original_rows),
            "reconstructed_row_count": len(rows) + 1,
            "original_column_count": (
                max(
                    (len(row) for row in original_rows),
                    default=0
                )
            ),
            "reconstructed_column_count": expected_columns,
            "numeric_values_modified_by_algorithm": False,
        }
    ]

    result["reconstruction_findings"] = []

    result["reconstruction_status"] = (
        "RECONSTRUCTED_MANUAL_COORDINATE"
    )

    return result


# ============================================================
# VALIDATION
# ============================================================

def validate_tables(
    original: list[dict[str, Any]],
    reconstructed: list[dict[str, Any]]
) -> dict[str, Any]:

    original_ids = [
        record.get("table_id")
        for record in original
    ]

    output_ids = [
        record.get("table_id")
        for record in reconstructed
    ]

    duplicate_original_ids = [
        table_id
        for table_id, count in Counter(original_ids).items()
        if count > 1
    ]

    duplicate_output_ids = [
        table_id
        for table_id, count in Counter(output_ids).items()
        if count > 1
    ]

    missing_ids = sorted(
        set(original_ids) - set(output_ids)
    )

    extra_ids = sorted(
        set(output_ids) - set(original_ids)
    )

    original_by_id = {
        record.get("table_id"): record
        for record in original
    }

    output_by_id = {
        record.get("table_id"): record
        for record in reconstructed
    }

    unchanged_non_targets = True

    for table_id in set(original_ids) - TARGET_IDS:
        if original_by_id[table_id] != output_by_id[table_id]:
            unchanged_non_targets = False
            break

    return {
        "total_original_tables": len(original),
        "total_output_tables": len(reconstructed),
        "target_ids": sorted(TARGET_IDS),
        "target_ids_found": sorted(
            set(original_ids) & TARGET_IDS
        ),
        "missing_target_ids": sorted(
            TARGET_IDS - set(original_ids)
        ),
        "duplicate_original_ids": duplicate_original_ids,
        "duplicate_output_ids": duplicate_output_ids,
        "missing_ids": missing_ids,
        "extra_ids": extra_ids,
        "non_target_tables_unchanged": unchanged_non_targets,
        "all_valid": (
            len(original) == len(reconstructed)
            and not duplicate_original_ids
            and not duplicate_output_ids
            and not missing_ids
            and not extra_ids
            and unchanged_non_targets
            and TARGET_IDS.issubset(set(original_ids))
        ),
    }


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    parser = argparse.ArgumentParser(
        description=(
            "Rekonstruksi tujuh tabel target berdasarkan "
            "koordinat PDF."
        )
    )

    parser.add_argument(
        "--dir",
        required=True,
        help="Folder yang berisi tables_clean.jsonl",
    )

    args = parser.parse_args()

    doc_dir = Path(args.dir)

    if not doc_dir.is_absolute():
        doc_dir = (Path.cwd() / doc_dir).resolve()

    input_path = doc_dir / "tables_clean.jsonl"
    output_path = doc_dir / "tables_reconstructed.jsonl"
    report_path = doc_dir / "reconstruction_report.json"

    print("=" * 65)
    print("REKONSTRUKSI TUJUH TABEL TARGET")
    print("=" * 65)

    print(f"Input: {input_path}")

    tables = read_jsonl(input_path)

    # Pastikan ID tabel tidak duplikat.
    ids = [table.get("table_id") for table in tables]

    if len(ids) != len(set(ids)):
        raise ValueError(
            "Ditemukan table_id duplikat pada file input."
        )

    available_ids = set(ids)
    missing_targets = TARGET_IDS - available_ids

    if missing_targets:
        raise ValueError(
            f"Tabel target tidak ditemukan: {sorted(missing_targets)}"
        )

    reconstructed = []
    processed_ids = []

    for table in tables:

        table_id = table.get("table_id")

        if table_id in TARGET_IDS:

            result = reconstruct_one_table(
                table,
                TABLE_SPECS[table_id]
            )

            processed_ids.append(table_id)

            print(
                f"[RECONSTRUCTED] "
                f"ID={table_id} "
                f"Halaman={table.get('page')} "
                f"Baris={len(TABLE_SPECS[table_id]['rows'])} "
                f"Kolom={len(TABLE_SPECS[table_id]['headers'])}"
            )

        else:
            # Tabel non-target dipertahankan tanpa perubahan.
            result = copy.deepcopy(table)

        reconstructed.append(result)

    # Validasi seluruh output.
    validation = validate_tables(
        tables,
        reconstructed
    )

    if not validation["all_valid"]:
        raise RuntimeError(
            "Validasi gagal. File output tidak akan ditulis.\n"
            + json.dumps(
                validation,
                ensure_ascii=False,
                indent=2
            )
        )

    # Simpan output.
    write_jsonl(output_path, reconstructed)

    summary = {
        "input_file": str(input_path),
        "output_file": str(output_path),
        "total_tables": len(tables),
        "reconstructed_count": len(processed_ids),
        "unchanged_count": len(tables) - len(processed_ids),
        "reconstructed_ids": sorted(processed_ids),
        "status": "SUCCESS",
        "validation": validation,
        "reconstruction_policy": (
            "Hanya tujuh tabel target direkonstruksi berdasarkan "
            "koordinat PDF. Tabel non-target disalin tanpa perubahan. "
            "Data asli clean_rows dan raw_rows dipertahankan."
        ),
    }

    report_path.write_text(
        json.dumps(
            summary,
            ensure_ascii=False,
            indent=2
        ),
        encoding="utf-8"
    )

    print("\n" + "=" * 65)
    print("HASIL REKONSTRUKSI")
    print("=" * 65)

    print(f"Total tabel       : {len(tables)}")
    print(f"Direkonstruksi    : {len(processed_ids)}")
    print(f"Tidak diubah      : {len(tables) - len(processed_ids)}")
    print(f"ID target         : {sorted(processed_ids)}")
    print(
        "Tabel non-target  : "
        f"{validation['non_target_tables_unchanged']}"
    )
    print(f"Validasi          : {validation['all_valid']}")

    print("\nFile berhasil disimpan:")
    print(f"JSONL  : {output_path}")
    print(f"REPORT : {report_path}")


if __name__ == "__main__":
    main()
