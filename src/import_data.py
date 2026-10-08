"""Smart ingestion and schema inference for CSV/Excel datasets."""

import re

import numpy as np
import pandas as pd

from src.preprocessing import COLUMNS, DataValidationError, clean_data
from src.auto_import import AUTO_NUMBER, parse_dates_auto, parse_numbers_auto

ALIASES = {
    "date": [
        "date", "tanggal", "tgl", "tanggal_harga", "tanggal_transaksi",
        "transaction_date", "order_date", "datetime", "timestamp", "waktu",
        "periode", "bulan", "waktu_pesanan_dibuat", "waktu_order", "order_time",
    ],
    "product": [
        "product", "produk", "nama_barang", "nama_produk", "barang", "komoditas",
        "product_name", "item", "commodity", "name", "title", "nama_komoditas",
        "item_name", "product_id", "sku", "product_categories",
        "kategori_produk", "kategori_barang", "nama_kategori_produk",
    ],
    "category": [
        "category", "kategori", "kategori_barang", "kategori_produk",
        "product_category",
    ],
    "previous_price": [
        "previous_price", "harga_sebelumnya", "harga_lalu", "last_price", "prev_price",
    ],
    "demand": [
        "demand", "permintaan", "jumlah_permintaan", "estimated_demand",
        "quantity", "qty", "jumlah", "jumlah_terjual", "terjual", "penjualan",
        "sales", "units_sold", "total_qty", "total_quantity",
    ],
    "price": [
        "price", "harga", "harga_barang", "harga_satuan", "unit_price",
        "selling_price", "price_idr", "harga_jual", "harga_total", "total_harga",
        "total_pembayaran", "total_payment", "nilai_transaksi", "transaction_value",
        "order_value",
    ],
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

DUPLICATE_POLICIES = {
    "Buang catatan yang berkonflik": "reject",
    "Gabungkan rata-rata (barang dan satuan sama)": "mean",
    "Gabungkan median (barang dan satuan sama)": "median",
}


def normalize_name(value) -> str:
    name = re.sub(r"([a-z])([A-Z])", r"\1_\2", str(value).strip())
    return re.sub(r"[^\w]+", "_", name.lower()).strip("_")


def _column_profile(series: pd.Series) -> dict:
    values = series.dropna().astype(str).str.strip()
    values = values[values.ne("")]
    sample = values.head(500)
    numeric = pd.to_numeric(sample, errors="coerce")
    numeric_rate = float(numeric.notna().mean()) if len(sample) else 0.0
    unique_rate = float(sample.nunique() / len(sample)) if len(sample) else 0.0
    text_rate = 1.0 - numeric_rate
    date_like_rate = 0.0
    if len(sample):
        parsed = pd.to_datetime(sample, errors="coerce", format="mixed", dayfirst=True)
        date_like_rate = float(parsed.notna().mean())
    return {
        "numeric_rate": numeric_rate,
        "text_rate": text_rate,
        "unique_rate": unique_rate,
        "date_like_rate": date_like_rate,
        "non_empty": len(sample),
    }


def _score_role(column, role: str, raw: pd.DataFrame) -> tuple[float, str]:
    normalized = normalize_name(column)
    profile = _column_profile(raw[column])
    aliases = {normalize_name(alias) for alias in ALIASES[role]}
    score = 0.0
    reasons = []

    if normalized in aliases:
        score += 100
        reasons.append("nama kolom cocok")
    elif role == "price" and re.search(
        r"(?:^|_)(price|harga|payment|pembayaran|sales|revenue|nilai|amount|total)(?:_|$)",
        normalized,
    ):
        score += 62
        reasons.append("nama kolom mengarah ke nilai harga/transaksi")
    elif role == "product" and re.search(
        r"(?:^|_)(product|produk|barang|item|komoditas|sku|category|kategori)(?:_|$)",
        normalized,
    ):
        score += 60
        reasons.append("nama kolom mengarah ke identitas produk")
    elif role == "demand" and re.search(
        r"(?:^|_)(qty|quantity|jumlah|demand|permintaan|sold|terjual|sales|volume)(?:_|$)",
        normalized,
    ):
        score += 60
        reasons.append("nama kolom mengarah ke kuantitas")

    excluded = {
        "order_id", "status_pesanan", "alasan_pembatalan", "opsi_pengiriman",
        "metode_pembayaran", "kota_kabupaten", "provinsi", "source_file",
    }
    if normalized in excluded:
        score -= 90
        reasons.append("terdeteksi sebagai kolom metadata/transaksi")

    if role == "date":
        score += profile["date_like_rate"] * 45
        if profile["date_like_rate"] >= 0.8:
            reasons.append("isi kolom berbentuk tanggal")
    elif role in {"price", "previous_price", "demand"}:
        score += profile["numeric_rate"] * 28
        if profile["numeric_rate"] >= 0.85:
            reasons.append("isi kolom numerik")
    elif role in {"product", "category"}:
        if profile["text_rate"] >= 0.8:
            score += 18
            reasons.append("isi kolom berupa teks")
        if 0.005 <= profile["unique_rate"] <= 0.75:
            score += 8

    if role == "product" and "id" in normalized:
        score -= 35
        reasons.append("lebih mirip ID daripada nama produk")
    if role == "price" and re.search(r"previous|prev|last|sebelumnya|lalu|diskon|discount|ongkos|shipping", normalized):
        score -= 40
        reasons.append("lebih mirip nilai pembanding/biaya lain")
    if role == "demand" and re.search(r"weight|berat|diskon|discount|ongkos|shipping", normalized):
        score -= 30
        reasons.append("lebih mirip berat/diskon/biaya")

    return score, "; ".join(reasons) or "inferensi dari struktur isi"


def suggest_mapping(columns) -> dict:
    """Map roles using names plus deterministic semantic fallbacks."""
    normalized = {normalize_name(column): column for column in columns}
    result = {}

    for role in COLUMNS:
        if role == "category":
            candidate_aliases = ALIASES[role]
        else:
            candidate_aliases = ALIASES[role]

        exact = [
            normalized[normalize_name(alias)]
            for alias in candidate_aliases
            if normalize_name(alias) in normalized
        ]
        result[role] = exact[0] if exact else None

    return result


def infer_mapping(raw: pd.DataFrame) -> dict:
    """Intelligently infer roles from headers and sample values."""
    result = {role: None for role in COLUMNS}

    used = set()
    candidates_by_role = {}
    for role in COLUMNS:
        ranked = []
        for column in raw.columns:
            score, reason = _score_role(column, role, raw)
            ranked.append((score, column, reason))
        candidates_by_role[role] = sorted(ranked, reverse=True, key=lambda item: item[0])

    # Strong semantic roles are assigned first.
    for role in ["date", "price", "product", "demand", "category", "previous_price"]:
        for score, column, _ in candidates_by_role[role]:
            if score < 35 or column in used:
                continue
            result[role] = column
            used.add(column)
            break

    # Transaction data special case: payment + quantity + order time.
    normalized = {normalize_name(c): c for c in raw.columns}
    transaction_date = next(
        (normalized[x] for x in ["waktu_pesanan_dibuat", "order_time", "order_date"] if x in normalized),
        None,
    )
    transaction_qty = next(
        (normalized[x] for x in ["total_qty", "total_quantity", "qty", "quantity"] if x in normalized),
        None,
    )
    transaction_payment = next(
        (normalized[x] for x in ["total_pembayaran", "total_payment", "order_value", "transaction_value"] if x in normalized),
        None,
    )
    if transaction_date and transaction_qty and transaction_payment:
        result["date"] = transaction_date
        result["demand"] = transaction_qty
        result["price"] = transaction_payment
        used.update({transaction_date, transaction_qty, transaction_payment})

        product_score = []
        for column in raw.columns:
            if column in {transaction_date, transaction_qty, transaction_payment}:
                continue
            score, reason = _score_role(column, "product", raw)
            product_score.append((score, column, reason))
        product_score.sort(reverse=True, key=lambda item: item[0])
        if product_score and product_score[0][0] >= 45:
            result["product"] = product_score[0][1]

    # Content-based date fallback for unknown headers.
    if result["date"] is None:
        ranked = candidates_by_role["date"]
        if ranked and ranked[0][0] >= 35:
            result["date"] = ranked[0][1]

    # Do not map total/category metadata as two roles if the same column was selected.
    for role in ["price", "product", "demand", "category", "previous_price"]:
        if result[role] in {result["date"], result["price"], result["product"], result["demand"]} and role not in {"price", "product", "demand"}:
            result[role] = None

    return result


def is_transaction_dataset(raw: pd.DataFrame, mapping: dict) -> bool:
    """Detect order-level data where total payment and quantity can form a unit-price proxy."""
    date_name = normalize_name(mapping.get("date") or "")
    price_name = normalize_name(mapping.get("price") or "")
    demand_name = normalize_name(mapping.get("demand") or "")
    return (
        bool(date_name and price_name and demand_name)
        and ("pesanan" in date_name or "order" in date_name or "transaksi" in date_name)
        and (
            "pembayaran" in price_name
            or "payment" in price_name
            or "transaction" in price_name
            or "order_value" in price_name
            or "nilai" in price_name
        )
        and ("qty" in demand_name or "quantity" in demand_name or "jumlah" in demand_name)
    )


def analyze_mapping(raw: pd.DataFrame, mapping: dict) -> dict:
    """Return explainable smart-mapping diagnostics for the UI."""
    details = []
    for role in ["date", "product", "category", "previous_price", "demand", "price"]:
        column = mapping.get(role)
        if column is None:
            details.append({
                "Peran": role,
                "Kolom": "—",
                "Keyakinan": "Tidak ditemukan",
                "Alasan": "Tidak ada kolom yang cukup kuat untuk peran ini.",
            })
            continue
        score, reason = _score_role(column, role, raw)
        confidence = "Tinggi" if score >= 90 else "Sedang" if score >= 55 else "Rendah"
        details.append({
            "Peran": role,
            "Kolom": str(column),
            "Keyakinan": confidence,
            "Alasan": reason,
        })

    transaction = is_transaction_dataset(raw, mapping)
    return {
        "dataset_type": "Transaksi / e-commerce" if transaction else "Data historis umum",
        "details": pd.DataFrame(details),
        "transaction_price_proxy": transaction,
    }


def import_dataset(
    raw: pd.DataFrame,
    mapping: dict,
    *,
    date_format: str = "%Y-%m-%d",
    number_format: str = NUMBER_FORMATS[0],
    product_name: str = "",
    category_name: str = "Umum",
    duplicate_policy: str = "reject",
) -> tuple[pd.DataFrame, dict]:
    """Normalize a mapped dataset and adapt transaction data when appropriate."""
    if date_format not in DATE_FORMATS.values() or number_format not in NUMBER_FORMATS:
        raise ValueError("Format tanggal atau angka tidak dikenali.")

    for column in ["date", "price"]:
        if mapping.get(column) is None:
            raise ValueError(
                f"Pemetaan untuk {column} belum ditemukan. "
                "Sistem membutuhkan tanggal dan nilai harga/penjualan untuk prediksi."
            )

    if mapping.get("product") is None and not product_name.strip():
        raise ValueError(
            "Pemetaan produk belum ditemukan. Upload dataset yang memiliki "
            "kolom produk/nama barang, atau gunakan dataset satu produk."
        )

    chosen = [mapping.get(column) for column in COLUMNS if mapping.get(column) is not None]
    if any(column not in raw.columns for column in chosen):
        raise ValueError("Pemetaan menunjuk kolom yang tidak tersedia pada dataset.")
    if len(chosen) != len(set(chosen)):
        raise ValueError(
            "Satu kolom dataset tidak boleh digunakan untuk dua peran. "
            "Periksa hasil Smart Mapping."
        )

    mapped = pd.DataFrame(index=raw.index)
    for column in COLUMNS:
        source = mapping.get(column)
        mapped[column] = raw[source] if source is not None else np.nan

    if mapping.get("product") is None:
        mapped["product"] = product_name.strip()
    if mapping.get("category") is None:
        mapped["category"] = category_name.strip() or "Umum"

    transaction_mode = is_transaction_dataset(raw, mapping)
    if transaction_mode:
        duplicate_policy = "mean"
        mapped["__source_quantity"] = mapped["demand"].copy()
        mapped["__source_payment"] = mapped["price"].copy()

    original = mapped[["date", "product", "price"]].copy()

    if date_format == "auto":
        mapped["date"] = parse_dates_auto(mapped["date"])
    elif date_format == "excel":
        serials = pd.to_numeric(mapped["date"], errors="coerce")
        mapped["date"] = pd.to_datetime(
            serials.where(serials.between(1, 100000)),
            unit="D",
            origin="1899-12-30",
            errors="coerce",
        ).dt.normalize()
    else:
        mapped["date"] = pd.to_datetime(
            mapped["date"].astype(str).str.strip(),
            format=date_format,
            errors="coerce",
        ).dt.normalize()

    for column in ["price", "previous_price", "demand"]:
        if number_format == AUTO_NUMBER:
            mapped[column] = parse_numbers_auto(
                mapped[column],
                str(mapping.get(column) or column),
            )
            continue
        numbers = (
            mapped[column]
            .fillna("")
            .astype(str)
            .str.strip()
            .str.replace(r"(?i)^rp\.?\s*", "", regex=True)
            .str.replace(r"\s+", "", regex=True)
        )
        if number_format == NUMBER_FORMATS[1]:
            numbers = numbers.str.replace(".", "", regex=False).str.replace(",", ".", regex=False)
        elif number_format == NUMBER_FORMATS[2]:
            numbers = numbers.str.replace(",", "", regex=False)
        mapped[column] = pd.to_numeric(numbers, errors="coerce")

    if transaction_mode:
        quantity = mapped["__source_quantity"].where(mapped["__source_quantity"] > 0)
        payment = mapped["__source_payment"].where(mapped["__source_payment"] > 0)
        mapped["price"] = payment / quantity
        mapped["demand"] = quantity

    def enrich_report(report):
        preview = report["rejected_preview"]
        report["rejected_preview"] = preview.join(original).rename(
            columns={
                "date": "Tanggal asli",
                "product": "Produk asli",
                "price": "Harga asli",
            }
        )

    if transaction_mode:
        # Multiple orders can exist for the same product/day. Aggregate them
        # into one daily observation so duplicate transaction timestamps do not
        # get rejected. The price is a quantity-weighted average.
        valid = mapped["date"].notna() & mapped["product"].astype(str).str.strip().ne("")
        working = mapped.loc[valid].copy()
        working["__weighted_value"] = working["price"] * working["demand"]

        def aggregate(group):
            qty = float(group["demand"].sum())
            weighted_value = float(group["__weighted_value"].sum())
            price = weighted_value / qty if qty > 0 else np.nan
            return pd.Series({
                "category": group["category"].iloc[0],
                "price": price,
                "demand": qty,
                "previous_price": np.nan,
            })

        aggregated = (
            working.groupby(["date", "product"], sort=False, dropna=False)
            .apply(aggregate, include_groups=False)
            .reset_index()
        )
        mapped = aggregated
        mapped = mapped[["date", "product", "category", "previous_price", "demand", "price"]]
        report = {
            "rows_in": len(raw),
            "rows_out_before_validation": len(mapped),
            "rows_removed": len(raw) - len(mapped),
            "transaction_dataset": True,
            "price_derivation": "Harga/unit estimasi = Total Pembayaran ÷ total_qty, lalu diagregasi per produk/hari.",
            "aggregated_rows": len(raw) - len(mapped),
            "previous_price_derived": True,
            "previous_demand_derived": True,
            "previous_price_missing": 0,
            "previous_demand_missing": 0,
            "demand_missing": int(mapped["demand"].isna().sum()),
            "invalid_dates": 0,
            "invalid_products": 0,
            "invalid_prices": 0,
            "conflicting_rows": 0,
            "rejected_preview": pd.DataFrame(),
        }
        try:
            frame, clean_report = clean_data(
                mapped,
                derive_previous_price=True,
                duplicate_policy="reject",
            )
        except DataValidationError as exc:
            enrich_report(exc.report)
            raise
        for key, value in report.items():
            if key == "rejected_preview":
                continue
            clean_report[key] = value
        enrich_report(clean_report)
        return frame, clean_report

    try:
        frame, report = clean_data(
            mapped,
            derive_previous_price=mapping.get("previous_price") is None,
            duplicate_policy=duplicate_policy,
        )
    except DataValidationError as exc:
        enrich_report(exc.report)
        raise

    enrich_report(report)
    report["category_defaulted"] = mapping.get("category") is None
    report["demand_added"] = mapping.get("demand") is None
    return frame, report
