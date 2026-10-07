"""Prediksi satu harga untuk periode berikutnya menggunakan Random Forest tersimpan."""

from datetime import date as Date

import numpy as np
import pandas as pd

from src.preprocessing import MAX_VALUE, make_features


def predict_price(
    bundle: dict,
    product: str,
    category: str,
    previous_price: float,
    date: str | Date,
    previous_demand: float | None = None,
) -> float:
    """Prediksi harga target dengan seluruh fitur tersedia sebelum periode target."""
    allowed = {(item["product"], item["category"]) for item in bundle["products"]}
    if (product, category) not in allowed:
        raise ValueError("Produk/kategori belum ada dalam data model. Tambahkan riwayat lalu latih ulang.")

    if not np.isfinite(previous_price) or not 0 < previous_price <= MAX_VALUE:
        raise ValueError("Harga sebelumnya harus positif dan maksimal 1 triliun.")

    if previous_demand is not None and (
        not np.isfinite(previous_demand) or not 0 <= previous_demand <= MAX_VALUE
    ):
        raise ValueError("Permintaan historis harus 0 atau lebih dan maksimal 1 triliun.")

    try:
        parsed_date = pd.to_datetime(str(date), format="%Y-%m-%d", errors="raise")
        if pd.isna(parsed_date):
            raise ValueError("Tanggal kosong.")
    except (ValueError, TypeError, OverflowError) as exc:
        raise ValueError("Tanggal prediksi tidak valid; gunakan YYYY-MM-DD.") from exc

    row = pd.DataFrame([{
        "date": parsed_date,
        "product": product,
        "category": category,
        "previous_price": float(previous_price),
        "previous_demand": np.nan if previous_demand is None else float(previous_demand),
    }])

    price = float(bundle["pipeline"].predict(make_features(row))[0])
    if not np.isfinite(price) or price <= 0:
        raise ValueError("Model menghasilkan harga tidak valid. Periksa dataset lalu latih ulang.")
    return price
