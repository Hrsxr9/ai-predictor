"""Validasi CSV dan feature engineering tanpa mempelajari data uji."""

from hashlib import sha256
import csv
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = ROOT / "data" / "historical_prices.csv"
DEFAULT_MODEL = ROOT / "models" / "model.pkl"
COLUMNS = ["date", "product", "category", "previous_price", "demand", "price"]
REQUIRED_COLUMNS = [column for column in COLUMNS if column != "demand"]
NUMERIC_FEATURES = ["year", "month", "day", "previous_price", "demand"]
CATEGORICAL_FEATURES = ["product", "category"]
MAX_ROWS = 50_000
MAX_BYTES = 10 * 1024 * 1024
MAX_VALUE = 1_000_000_000_000


class DataValidationError(ValueError):
    """Sertakan rincian agar UI dapat menjelaskan mengapa tidak ada data tersisa."""

    def __init__(self, message, report):
        super().__init__(message)
        self.report = report


def read_csv(source: Path | str | bytes) -> pd.DataFrame:
    """Baca CSV UTF-8 dengan pemisah koma, titik koma, tab, atau pipa."""
    if not isinstance(source, bytes):
        source = Path(source)
        if source.stat().st_size > MAX_BYTES:
            raise ValueError("CSV terlalu besar. Maksimum 10 MB.")
        source = source.read_bytes()
    if len(source) > MAX_BYTES:
        raise ValueError("CSV terlalu besar. Maksimum 10 MB.")
    try:
        contents = source.decode("utf-8-sig")
        try:
            separator = csv.Sniffer().sniff(contents[:8192], delimiters=",;\t|").delimiter
        except csv.Error:
            header = next((line for line in contents.splitlines() if line.strip()), "")
            separator = max(",;\t|", key=header.count) if header else ","
        frame = pd.read_csv(
            StringIO(contents), sep=separator, dtype=str, keep_default_na=False,
            nrows=MAX_ROWS + 1,
        )
    except (UnicodeError, pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
        raise ValueError("CSV tidak dapat dibaca. Gunakan UTF-8, header kolom, dan pemisah koma/titik koma/tab/pipa yang konsisten.") from exc
    if len(frame) > MAX_ROWS:
        raise ValueError("Maksimum 50.000 baris untuk aplikasi lokal versi ini.")
    return frame


def clean_data(raw: pd.DataFrame, derive_previous_price: bool = False,
               duplicate_policy: str = "reject") -> tuple[pd.DataFrame, dict]:
    """Bersihkan aturan tetap; imputasi median baru dilakukan saat training."""
    frame = raw.copy()
    if duplicate_policy not in {"reject", "mean", "median"}:
        raise ValueError("Pilihan penanganan catatan ganda tidak dikenali.")
    frame.columns = [str(column).strip().lower() for column in frame.columns]
    if frame.columns.duplicated().any():
        raise ValueError("Nama kolom tidak boleh duplikat setelah spasi dihapus.")
    missing = sorted(set(REQUIRED_COLUMNS) - set(frame.columns))
    if missing:
        raise ValueError("Kolom wajib belum tersedia: " + ", ".join(missing))
    report = {"rows_in": len(frame), "demand_added": "demand" not in frame}
    if len(frame) > MAX_ROWS:
        raise ValueError("Maksimum 50.000 baris.")
    if "demand" not in frame:
        frame["demand"] = np.nan
    frame = frame[COLUMNS].copy()
    for column in ["product", "category"]:
        frame[column] = frame[column].fillna("").astype(str).str.strip()
    report["category_filled"] = int(frame["category"].eq("").sum())
    frame["category"] = frame["category"].replace("", "Tidak diketahui")

    if pd.api.types.is_datetime64_any_dtype(frame["date"]):
        dates = frame["date"].dt.strftime("%Y-%m-%d")
    else:
        dates = frame["date"].fillna("").astype(str).str.strip()
    frame["date"] = pd.to_datetime(
        dates.where(dates.str.fullmatch(r"\d{4}-\d{2}-\d{2}", na=False)),
        format="%Y-%m-%d", errors="coerce",
    )
    for column in ["previous_price", "demand", "price"]:
        numbers = pd.to_numeric(frame[column], errors="coerce").astype(float)
        valid = np.isfinite(numbers) & numbers.le(MAX_VALUE)
        valid &= numbers.ge(0) if column == "demand" else numbers.gt(0)
        frame[column] = numbers.where(valid, np.nan)

    checks = {"Tanggal tidak terbaca": frame["date"].isna(),
              "Produk kosong": frame["product"].eq(""), "Harga tidak valid": frame["price"].isna()}
    report["invalid_dates"] = int(checks["Tanggal tidak terbaca"].sum())
    report["invalid_products"] = int(checks["Produk kosong"].sum())
    report["invalid_prices"] = int(checks["Harga tidak valid"].sum())
    invalid = checks["Tanggal tidak terbaca"] | checks["Produk kosong"] | checks["Harga tidak valid"]
    reasons = pd.Series("", index=frame.index)
    for label, mask in checks.items():
        reasons.loc[mask] += label + "; "
    report["rejected_preview"] = pd.DataFrame({
        "Baris CSV": np.arange(len(frame)) + 2, "Alasan": reasons.str.rstrip("; "),
    }, index=frame.index).loc[invalid].head(10)
    report["invalid_rows"] = int(invalid.sum())
    frame = frame.loc[~invalid].copy()
    report["duplicate_rows"] = int(frame.duplicated().sum())
    frame = frame.drop_duplicates()
    conflicts = frame.duplicated(["date", "product"], keep=False)
    report["conflicting_rows"] = 0
    report["aggregated_rows"] = 0
    if duplicate_policy == "reject":
        report["conflicting_rows"] = int(conflicts.sum())
        frame = frame.loc[~conflicts]
    elif conflicts.any():
        if (frame.groupby(["date", "product"])["category"].nunique() > 1).any():
            raise ValueError("Kategori berbeda untuk produk-tanggal yang sama. Perbaiki nama produk/pemetaan sebelum menggabungkan harga.")
        before = len(frame)
        frame = frame.groupby(["date", "product"], as_index=False).agg({
            "category": "first", "price": duplicate_policy, "demand": duplicate_policy,
            "previous_price": duplicate_policy,
        })
        report["aggregated_rows"] = before - len(frame)
        derive_previous_price = True
    frame = frame.sort_values(["date", "product"]).reset_index(drop=True)
    if derive_previous_price:
        frame["previous_price"] = frame.groupby("product", sort=False)["price"].shift(1)
    report["previous_price_derived"] = derive_previous_price
    report["previous_price_missing"] = int(frame["previous_price"].isna().sum())
    report["demand_missing"] = int(frame["demand"].isna().sum())
    report["rows_out"] = len(frame)
    report["rows_removed"] = len(raw) - len(frame)
    if frame.empty:
        raise DataValidationError(
            f"Tidak ada baris valid dari {len(raw)} baris: {report['invalid_dates']} tanggal tidak terbaca, "
            f"{report['invalid_products']} produk kosong, {report['invalid_prices']} harga tidak valid, "
            f"{report['conflicting_rows']} baris konflik produk-tanggal. "
            "Periksa contoh baris di bawah. Untuk konflik dari barang dan satuan yang sama, "
            "pilih gabungkan rata-rata/median; jika barang berbeda, perbaiki kolom produk.", report,
        )
    return frame, report


def make_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Fungsi yang sama dipanggil saat training, pengujian, dan prediksi."""
    result = frame[["product", "category", "previous_price", "demand"]].copy()
    dates = pd.to_datetime(frame["date"], errors="raise")
    result["year"] = dates.dt.year
    result["month"] = dates.dt.month
    result["day"] = dates.dt.day
    return result[NUMERIC_FEATURES + CATEGORICAL_FEATURES]


def dataset_fingerprint(frame: pd.DataFrame) -> str:
    """Model lama tidak digunakan untuk dataset yang isinya sudah berubah."""
    contents = frame[COLUMNS].to_csv(index=False, date_format="%Y-%m-%d", float_format="%.12g")
    return sha256(contents.encode("utf-8")).hexdigest()


def chronological_split(frame: pd.DataFrame, test_fraction: float = 0.2):
    if not 0.1 <= test_fraction <= 0.4:
        raise ValueError("Porsi data uji harus antara 10% dan 40%.")
    dates = np.sort(frame["date"].unique())
    if len(frame) < 40 or len(dates) < 10:
        raise ValueError("Training memerlukan minimal 40 baris valid dan 10 tanggal berbeda.")
    boundary = dates[int(np.floor(len(dates) * (1 - test_fraction)))]
    train = frame.loc[frame["date"] < boundary].copy()
    test = frame.loc[frame["date"] >= boundary].copy()
    if len(train) < 20 or len(test) < 5:
        raise ValueError("Pembagian waktu menghasilkan terlalu sedikit data: perlu ≥20 train dan ≥5 test.")
    if train["previous_price"].notna().sum() == 0:
        raise ValueError("Isi minimal satu previous_price yang valid pada periode training.")
    return train, test


def rupiah(value: float) -> str:
    return "Rp " + f"{value:,.0f}".replace(",", ".")
