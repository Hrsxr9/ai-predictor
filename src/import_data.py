"""Pemetaan CSV pengguna ke schema internal tanpa menebak data yang tidak ada."""

import re

import numpy as np
import pandas as pd

from src.preprocessing import COLUMNS, DataValidationError, clean_data
from src.auto_import import AUTO_NUMBER, parse_dates_auto, parse_numbers_auto

ALIASES = {
    "date": ["date", "tanggal", "tgl", "tanggal_harga", "tanggal_transaksi", "transaction_date", "order_date", "datetime", "timestamp", "waktu", "periode", "bulan"],
    "product": ["product", "produk", "nama_barang", "nama_produk", "barang", "komoditas", "product_name", "item", "commodity", "name", "title", "nama_komoditas", "item_name", "product_id", "sku"],
    "category": ["category", "kategori", "kategori_barang", "kategori_produk", "product_category"],
    "previous_price": ["previous_price", "harga_sebelumnya", "harga_lalu", "last_price", "prev_price"],
    "demand": ["demand", "permintaan", "jumlah_permintaan", "estimated_demand", "quantity", "qty", "jumlah", "jumlah_terjual", "terjual", "penjualan", "sales", "units_sold"],
    "price": ["price", "harga", "harga_barang", "harga_satuan", "unit_price", "selling_price", "price_idr"],
}
DATE_FORMATS = {
    "YYYY-MM-DD (2026-01-31)": "%Y-%m-%d",
    "DD/MM/YYYY (31/01/2026)": "%d/%m/%Y",
    "DD-MM-YYYY (31-01-2026)": "%d-%m-%Y",
    "MM/DD/YYYY (01/31/2026)": "%m/%d/%Y",
    "YYYY/MM/DD (2026/01/31)": "%Y/%m/%d",
    "YYYY-MM-DD HH:MM:SS (2026-01-31 08:00:00)": "%Y-%m-%d %H:%M:%S",
    "Otomatis (deteksi dari isi)": "auto",
    "Tanggal Excel (contoh 46053)": "excel",
}
NUMBER_FORMATS = [
    "Angka standar: 15500 atau 15500.50",
    "Indonesia: 15.500 atau 15.500,50",
    "Internasional: 15,500 atau 15,500.50",
    AUTO_NUMBER,
]
DUPLICATE_POLICIES = {"Buang catatan yang berkonflik": "reject",
                      "Gabungkan rata-rata (barang dan satuan sama)": "mean",
                      "Gabungkan median (barang dan satuan sama)": "median"}


def suggest_mapping(columns) -> dict:
    """Saran hanya mencocokkan nama; pengguna memeriksa makna kolomnya."""
    normalized = {}
    for column in columns:
        name = re.sub(r"([a-z])([A-Z])", r"\1_\2", str(column).strip())
        name = re.sub(r"[^\w]+", "_", name.lower()).strip("_")
        normalized.setdefault(name, column)
    result = {}
    for target, aliases in ALIASES.items():
        matches = [normalized[alias] for alias in aliases if alias in normalized]
        if not matches and target == "price":
            matches = [column for name, column in normalized.items()
                       if re.search(r"(?:^|_)(price|harga)(?:_|$)", name)
                       and not re.search(r"previous|prev|last|sebelumnya|lalu|change|selisih", name)]
        result[target] = normalized.get(target) or (matches[0] if len(matches) == 1 else None)
    return result


def infer_mapping(raw: pd.DataFrame) -> dict:
    """Nama kolom sebagai dasar; isi berbentuk tanggal membantu header tidak dikenal."""
    mapping = suggest_mapping(raw.columns)
    if mapping["date"] is None:
        candidates = []
        for column in raw.columns:
            if column in mapping.values():
                continue
            values = raw[column].dropna().astype(str).str.strip()
            values = values.loc[values.ne("")].head(500)
            if len(values) and values.str.match(r"^(?:\d{4}[-/]\d{1,2}[-/]\d{1,2}|\d{1,2}[-/]\d{1,2}[-/]\d{4})(?:\b|T)").mean() >= 0.9:
                candidates.append(column)
        if len(candidates) == 1:
            mapping["date"] = candidates[0]
    return mapping


def import_dataset(raw: pd.DataFrame, mapping: dict, *,
                   date_format: str = "%Y-%m-%d", number_format: str = NUMBER_FORMATS[0],
                   product_name: str = "", category_name: str = "Umum",
                   duplicate_policy: str = "reject") -> tuple[pd.DataFrame, dict]:
    """Normalisasi pilihan eksplisit pengguna, lalu jalankan validasi bersama."""
    if date_format not in DATE_FORMATS.values() or number_format not in NUMBER_FORMATS:
        raise ValueError("Format tanggal atau angka tidak dikenali.")
    for column in ["date", "price"]:
        if mapping.get(column) is None:
            raise ValueError(f"Pilih kolom {column}. CSV harus memiliki tanggal dan harga historis yang nyata.")
    if mapping.get("product") is None and not product_name.strip():
        raise ValueError("Pilih kolom produk, atau isi nama barang jika seluruh CSV memang berisi satu produk.")
    chosen = [mapping.get(column) for column in COLUMNS if mapping.get(column) is not None]
    if any(column not in raw.columns for column in chosen):
        raise ValueError("Pilihan kolom tidak tersedia di CSV ini. Periksa pemetaan kembali.")
    if len(chosen) != len(set(chosen)):
        raise ValueError("Satu kolom CSV tidak boleh dipakai untuk dua peran. Harga aktual dan harga sebelumnya harus berbeda.")

    mapped = pd.DataFrame(index=raw.index)
    for column in COLUMNS:
        source = mapping.get(column)
        mapped[column] = raw[source] if source is not None else np.nan
    if mapping.get("product") is None:
        mapped["product"] = product_name.strip()
    if mapping.get("category") is None:
        mapped["category"] = category_name.strip() or "Umum"

    original = mapped[["date", "product", "price"]].copy()
    if date_format == "auto":
        mapped["date"] = parse_dates_auto(mapped["date"])
    elif date_format == "excel":
        serials = pd.to_numeric(mapped["date"], errors="coerce")
        mapped["date"] = pd.to_datetime(serials.where(serials.between(1, 100000)), unit="D", origin="1899-12-30", errors="coerce").dt.normalize()
    else:
        mapped["date"] = pd.to_datetime(
            mapped["date"].astype(str).str.strip(), format=date_format, errors="coerce",
        ).dt.normalize()
    for column in ["price", "previous_price", "demand"]:
        if number_format == AUTO_NUMBER:
            mapped[column] = parse_numbers_auto(mapped[column], str(mapping.get(column) or column))
            continue
        numbers = mapped[column].fillna("").astype(str).str.strip()
        numbers = numbers.str.replace(r"(?i)^rp\.?\s*", "", regex=True).str.replace(r"\s+", "", regex=True)
        if number_format == NUMBER_FORMATS[1]:
            numbers = numbers.str.replace(".", "", regex=False).str.replace(",", ".", regex=False)
        elif number_format == NUMBER_FORMATS[2]:
            numbers = numbers.str.replace(",", "", regex=False)
        mapped[column] = numbers
    def enrich_report(report):
        preview = report["rejected_preview"]
        report["rejected_preview"] = preview.join(original).rename(columns={"date": "Tanggal asli", "product": "Produk asli", "price": "Harga asli"})

    try:
        frame, report = clean_data(mapped, derive_previous_price=mapping.get("previous_price") is None,
                                   duplicate_policy=duplicate_policy)
    except DataValidationError as exc:
        enrich_report(exc.report)
        raise
    enrich_report(report)
    report["category_defaulted"] = mapping.get("category") is None
    report["demand_added"] = mapping.get("demand") is None
    return frame, report
