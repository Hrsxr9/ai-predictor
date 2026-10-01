"""Buat data demonstrasi sintetis yang dapat direproduksi, bukan harga pasar asli."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.preprocessing import COLUMNS, DEFAULT_DATA


def generate_dataset(seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    products = [
        ("Beras medium (kg)", "Sembako", 12_000, 900, 0.035),
        ("Gula pasir (kg)", "Sembako", 14_000, 600, 0.040),
        ("Minyak goreng (liter)", "Sembako", 16_000, 650, 0.055),
        ("Telur ayam (kg)", "Protein", 24_000, 500, 0.060),
        ("Daging ayam (kg)", "Protein", 32_000, 400, 0.075),
        ("Cabai merah (kg)", "Sayuran", 38_000, 300, 0.180),
    ]
    rows = []
    for product, category, base, base_demand, volatility in products:
        previous = float(base)
        phase = rng.uniform(0, np.pi)
        for index, date in enumerate(pd.date_range("2020-01-01", periods=80, freq="MS")):
            season = np.sin(2 * np.pi * (date.month - 1) / 12 + phase)
            demand = int(max(1, base_demand * (1 + 0.18 * season + rng.normal(0, 0.055))))
            ratio = demand / base_demand - 1
            equilibrium = base * (
                1 + 0.002 * index + volatility * season + 0.22 * ratio + 0.35 * max(ratio, 0) ** 2
            )
            price = 0.65 * previous + 0.35 * equilibrium + rng.normal(0, base * volatility * 0.12)
            price = float(max(500, round(price / 50) * 50))
            rows.append([date, product, category, previous, demand, price])
            previous = price
    return pd.DataFrame(rows, columns=COLUMNS).sort_values(["date", "product"]).reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser(description="Buat 480 baris data sintetis lokal.")
    parser.add_argument("--output", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--force", action="store_true", help="Izinkan menimpa file keluaran yang ada.")
    args = parser.parse_args()
    if args.output.exists() and not args.force:
        parser.error("File sudah ada. Pilih --output lain atau gunakan --force.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    generate_dataset().to_csv(args.output, index=False, date_format="%Y-%m-%d")
    print(f"480 baris sintetis disimpan: {args.output}")


if __name__ == "__main__":
    main()
