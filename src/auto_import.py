"""Deteksi format lokal dengan aturan transparan; tidak memakai API AI."""

import re

import numpy as np
import pandas as pd

AUTO_NUMBER = "Otomatis (deteksi per kolom)"
MONTHS = {"januari": "January", "februari": "February", "maret": "March", "april": "April",
          "mei": "May", "juni": "June", "juli": "July", "agustus": "August", "agt": "August",
          "september": "September", "oktober": "October", "okt": "October",
          "november": "November", "desember": "December", "des": "December"}


def parse_dates_auto(values: pd.Series) -> pd.Series:
    strings = values.fillna("").astype(str).str.strip()
    strings = strings.str.replace(r"[A-Za-z]+", lambda match: MONTHS.get(match.group().lower(), match.group()), regex=True)
    strings = strings.str.replace(r"^(\d{4}[-/]\d{2}[-/]\d{2})[T ].*$", r"\1", regex=True)
    regional = strings.str.extract(r"^(\d{1,2})[/-](\d{1,2})[/-]\d{4}(?:\s.*)?$").astype(float)
    day_evidence = ((regional[0] > 12) & regional[0].le(31) & regional[1].between(1, 12)).any()
    month_evidence = ((regional[1] > 12) & regional[1].le(31) & regional[0].between(1, 12)).any()
    ambiguous = regional[0].between(1, 12) & regional[1].between(1, 12) & regional[0].ne(regional[1])
    if ambiguous.any() and (day_evidence == month_evidence):
        examples = ", ".join(strings.loc[ambiguous].head(3))
        raise ValueError(f"Urutan hari/bulan belum pasti ({examples}). Pilih DD/MM/YYYY atau MM/DD/YYYY pada Format tanggal.")
    if day_evidence and month_evidence:
        raise ValueError("CSV mencampur urutan hari/bulan dan bulan/hari. Samakan format tanggal terlebih dahulu.")
    compact = strings.str.fullmatch(r"\d{8}")
    monthly = strings.str.fullmatch(r"\d{4}-\d{2}")
    strings = strings.mask(monthly, strings + "-01")
    numeric = strings.str.fullmatch(r"[+-]?\d+(?:\.\d+)?")
    dates = pd.to_datetime(strings.mask(numeric), format="mixed", dayfirst=bool(day_evidence), errors="coerce")
    dates.loc[compact] = pd.to_datetime(strings.loc[compact], format="%Y%m%d", errors="coerce")
    return dates.dt.normalize()


def parse_numbers_auto(values: pd.Series, label: str) -> pd.Series:
    original = values.fillna("").astype(str).str.strip()
    tokens = original.str.replace(r"(?i)^rp\.?\s*", "", regex=True).str.replace(r"\s+", "", regex=True)
    styles, ambiguous, parsed = set(), [], {}
    for token in tokens.unique():
        if not token:
            parsed[token] = np.nan
            continue
        dots, commas = token.count("."), token.count(",")
        if dots and commas:
            decimal = "." if token.rfind(".") > token.rfind(",") else ","
            thousands = "," if decimal == "." else "."
            pattern = rf"[+-]?(?:\d{{1,3}}(?:{re.escape(thousands)}\d{{3}})+|\d+){re.escape(decimal)}\d+"
            if not re.fullmatch(pattern, token):
                parsed[token] = np.nan
                continue
            styles.add("intl" if decimal == "." else "id")
            parsed[token] = float(token.replace(thousands, "").replace(decimal, "."))
        elif dots + commas > 1:
            separator = "." if dots else ","
            if re.fullmatch(rf"[+-]?\d{{1,3}}(?:{re.escape(separator)}\d{{3}})+", token):
                styles.add("id" if separator == "." else "intl")
                parsed[token] = float(token.replace(separator, ""))
            else:
                parsed[token] = np.nan
        elif dots + commas == 1 and "e" not in token.lower():
            separator = "." if dots else ","
            parts = token.lstrip("+-").split(separator)
            if not all(part.isdigit() for part in parts):
                parsed[token] = pd.to_numeric(token, errors="coerce")
            elif 1 <= len(parts[0]) <= 3 and len(parts[1]) == 3 and int(parts[0]) != 0:
                ambiguous.append(token)
            else:
                styles.add("intl" if separator == "." else "id")
                parsed[token] = float(token.replace(",", "."))
        else:
            parsed[token] = pd.to_numeric(token, errors="coerce")
    if not styles and original.str.match(r"(?i)^rp\.?\s*").any():
        styles.add("id")
    if ambiguous and len(styles) != 1:
        raise ValueError(f"Format angka kolom {label} belum pasti ({', '.join(ambiguous[:3])}). Pilih format Indonesia atau Internasional/Angka standar.")
    for token in ambiguous:
        if "id" in styles:
            parsed[token] = float(token.replace(".", "").replace(",", "."))
        else:
            parsed[token] = float(token.replace(",", ""))
    return tokens.map(parsed).astype(float)
