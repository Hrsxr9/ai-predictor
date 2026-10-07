"""Streamlit UI untuk AI Price Predictor."""

from hashlib import sha256
from io import BytesIO
from pathlib import Path

import pandas as pd
import streamlit as st

from src.auto_import import AUTO_NUMBER
from src.generate_data import generate_dataset
from src.import_data import import_dataset, infer_mapping
from src.preprocessing import (
    DEFAULT_DATA,
    clean_data,
    dataset_fingerprint,
    read_csv,
    rupiah,
)
from src.predict import predict_price
from src.train import (
    BASELINE_NAME,
    LR_NAME,
    RF_NAME,
    load_model,
    save_model,
    train_models,
)

st.set_page_config(page_title="AI Price Predictor", page_icon="📈", layout="wide")


def ensure_demo_data():
    DEFAULT_DATA.parent.mkdir(parents=True, exist_ok=True)
    if not DEFAULT_DATA.exists():
        generate_dataset().to_csv(
            DEFAULT_DATA,
            index=False,
            date_format="%Y-%m-%d",
        )


def read_uploaded_table(uploaded):
    """Read CSV or Excel upload and choose the most useful worksheet."""
    suffix = Path(uploaded.name).suffix.lower()
    raw_bytes = uploaded.getvalue()

    if suffix == ".csv":
        return read_csv(raw_bytes), "CSV"

    if suffix in {".xlsx", ".xls"}:
        try:
            sheets = pd.read_excel(BytesIO(raw_bytes), sheet_name=None)
        except ImportError as exc:
            raise ValueError(
                "Pembacaan Excel memerlukan dependency Excel. Jalankan "
                "python -m pip install -r requirements.txt."
            ) from exc
        except Exception as exc:
            raise ValueError(f"File Excel tidak dapat dibaca: {exc}") from exc

        candidates = []
        for sheet_name, sheet in sheets.items():
            if sheet is None or sheet.empty:
                continue
            usable = sheet.dropna(how="all").copy()
            candidates.append(
                (len(usable.columns), len(usable), str(sheet_name), usable)
            )

        if not candidates:
            raise ValueError("Workbook Excel tidak memiliki sheet berisi data.")

        _, _, sheet_name, selected = max(
            candidates,
            key=lambda item: (item[0], item[1]),
        )
        return selected.astype(object), f"Excel · sheet '{sheet_name}'"

    raise ValueError("Format file tidak didukung. Gunakan CSV, XLSX, atau XLS.")


def load_active_data(uploaded=None):
    if uploaded is None:
        for key in (
            "uploaded_key",
            "uploaded_frame",
            "uploaded_report",
            "uploaded_name",
        ):
            st.session_state.pop(key, None)

        frame, report = clean_data(pd.read_csv(DEFAULT_DATA))
        return frame, report, "Dataset contoh"

    raw_bytes = uploaded.getvalue()
    upload_key = sha256(raw_bytes).hexdigest()

    if (
        st.session_state.get("uploaded_key") == upload_key
        and st.session_state.get("uploaded_frame") is not None
    ):
        return (
            st.session_state["uploaded_frame"],
            st.session_state["uploaded_report"],
            st.session_state["uploaded_name"],
        )

    try:
        raw, detected_format = read_uploaded_table(uploaded)
    except ValueError as exc:
        st.error(str(exc))
        return None, None, uploaded.name

    mapping = infer_mapping(raw)

    required_missing = [
        key for key in ("date", "product", "price")
        if mapping.get(key) is None
    ]

    if required_missing:
        st.warning(
            "Pemetaan otomatis belum menemukan: "
            + ", ".join(required_missing)
            + ". Periksa nama kolom CSV atau gunakan kolom yang jelas "
              "seperti tanggal, produk, dan harga."
        )
        st.subheader("Kolom CSV terdeteksi")
        st.dataframe(
            pd.DataFrame(
                {
                    "Peran sistem": list(mapping.keys()),
                    "Kolom terdeteksi": [
                        mapping.get(k) for k in mapping.keys()
                    ],
                }
            ),
            use_container_width=True,
            hide_index=True,
        )
        st.dataframe(raw.head(10), use_container_width=True, hide_index=True)
        return None, None, uploaded.name

    try:
        frame, report = import_dataset(
            raw,
            mapping,
            date_format="auto",
            number_format=AUTO_NUMBER,
            duplicate_policy="reject",
        )
    except ValueError as exc:
        st.error(f"CSV terbaca, tetapi validasi gagal: {exc}")
        st.info(
            "Periksa header dan isi CSV. Sistem sudah mencoba "
            "pemetaan kolom dan format angka/tanggal secara otomatis."
        )
        st.dataframe(raw.head(10), use_container_width=True, hide_index=True)
        return None, None, uploaded.name

    st.session_state["uploaded_key"] = upload_key
    st.session_state["uploaded_frame"] = frame
    st.session_state["uploaded_report"] = report
    st.session_state["uploaded_name"] = uploaded.name
    st.session_state["uploaded_format"] = detected_format
    st.session_state["mapping"] = mapping
    return frame, report, uploaded.name


def next_month_from(value):
    return pd.Timestamp(value).normalize() + pd.offsets.MonthBegin(1)


def main():
    ensure_demo_data()

    st.title("AI Price Predictor")
    st.caption(
        "Prediksi harga periode berikutnya berdasarkan pola data historis · Full Local"
    )

    with st.sidebar:
        st.header("Dataset")
        source = st.radio("Sumber", ["Dataset contoh", "Upload CSV"])
        uploaded = st.file_uploader(
            "CSV",
            type=["csv", "xlsx", "xls"],
            disabled=source != "Upload CSV",
            help=(
                "CSV hingga 10 MB / 50.000 baris; Excel XLSX/XLS juga didukung. "
                "Untuk Excel, sheet paling layak dipilih otomatis."
            ),
        )

        if source == "Upload CSV" and uploaded is not None:
            st.success("Pemetaan CSV: otomatis")
            mapping = st.session_state.get("mapping")
            if mapping:
                st.caption(
                    "→ "
                    + " · ".join(
                        f"{k}: {v}"
                        for k, v in mapping.items()
                        if v is not None
                    )
                )

        st.divider()
        page = st.radio(
            "Menu",
            [
                "Dashboard",
                "Dataset",
                "Training Model",
                "Prediksi Periode Berikutnya",
                "Model Comparison",
            ],
        )

    active = load_active_data(
        uploaded if source == "Upload CSV" else None
    )
    if active[0] is None:
        st.info(
            "Upload CSV sudah dibaca. Setelah pemetaan otomatis berhasil, "
            "menu akan menggunakan dataset tersebut."
        )
        return

    frame, report, source_name = active
    fingerprint = dataset_fingerprint(frame)

    if st.session_state.get("dataset_hash") != fingerprint:
        st.session_state["dataset_hash"] = fingerprint
        st.session_state.pop("bundle", None)
        try:
            path = Path("models/model.pkl")
            if path.exists():
                st.session_state["bundle"] = load_model(
                    path,
                    fingerprint,
                )
        except Exception:
            pass

    bundle = st.session_state.get("bundle")

    if page == "Dashboard":
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Data valid", f"{len(frame):,}")
        c2.metric("Produk", frame["product"].nunique())
        c3.metric("Harga minimum", rupiah(frame["price"].min()))
        c4.metric("Harga maksimum", rupiah(frame["price"].max()))

        st.subheader("Tren harga")
        product = st.selectbox(
            "Produk",
            sorted(frame["product"].unique()),
        )
        history = (
            frame[frame["product"] == product]
            .set_index("date")[["price"]]
        )
        st.line_chart(
            history.rename(columns={"price": "Harga (Rp)"})
        )
        st.info(
            "Alur AI: data historis → preprocessing → feature engineering "
            "→ training → evaluasi → prediksi periode berikutnya."
        )

    elif page == "Dataset":
        st.subheader("Dataset aktif")
        st.write(f"Sumber: **{source_name}**")

        if source == "Upload CSV" and st.session_state.get("mapping"):
            with st.expander("Pemetaan otomatis"):
                st.json(st.session_state["mapping"])

        st.dataframe(
            frame,
            use_container_width=True,
            hide_index=True,
        )

        st.write({
            "Baris masuk": report.get("rows_in"),
            "Baris valid": report.get("rows_out"),
            "Baris dibuang": report.get("rows_removed"),
            "Previous price kosong": report.get("previous_price_missing"),
            "Previous demand kosong": report.get("previous_demand_missing"),
            "Previous demand kosong": report.get("previous_demand_missing"),
            "Demand kosong": report.get("demand_missing"),
            "Format sumber": st.session_state.get("uploaded_format", "CSV"),
        })

    elif page == "Training Model":
        st.subheader("Training & evaluasi")
        st.write(
            "Data dibagi berdasarkan urutan waktu. Random Forest dan "
            "Linear Regression dibandingkan dengan baseline harga sebelumnya."
        )

        if st.button("Latih model", type="primary"):
            with st.spinner("Training..."):
                bundle = train_models(frame)
                Path("models").mkdir(exist_ok=True)
                save_model(bundle, Path("models/model.pkl"))
                st.session_state["bundle"] = bundle

            st.success(
                "Model selesai dilatih. Random Forest disimpan untuk "
                "prediksi periode berikutnya."
            )

        if bundle:
            st.dataframe(
                bundle["metrics"],
                use_container_width=True,
                hide_index=True,
            )
            st.caption(
                f"Train {bundle['split']['train_rows']} baris · "
                f"Test {bundle['split']['test_rows']} baris"
            )

    elif page == "Prediksi Periode Berikutnya":
        st.subheader("Prediksi harga periode berikutnya")

        if not bundle:
            st.info("Latih model terlebih dahulu.")
            return

        product = st.selectbox(
            "Produk",
            sorted(frame["product"].unique()),
        )
        product_history = (
            frame[frame["product"] == product]
            .sort_values("date")
            .copy()
        )
        latest = product_history.iloc[-1]

        category = str(latest["category"])
        latest_date = pd.Timestamp(latest["date"]).normalize()
        target_date = next_month_from(latest_date)

        previous_price = float(latest["price"])
        previous_demand = (
            None
            if pd.isna(latest["demand"])
            else float(latest["demand"])
        )

        st.write({
            "Produk": product,
            "Kategori": category,
            "Data terakhir": latest_date.strftime("%d %B %Y"),
            "Harga terakhir": rupiah(previous_price),
            "Demand terakhir": (
                "Tidak tersedia"
                if previous_demand is None
                else f"{previous_demand:,.0f}"
            ),
            "Periode yang diprediksi": target_date.strftime("%B %Y"),
        })

        st.info(
            "Sistem otomatis memakai harga dan demand terakhir yang sudah "
            "tersedia. Pengguna tidak perlu memasukkan data masa depan."
        )

        if st.button("Prediksi Harga Berikutnya", type="primary"):
            value = predict_price(
                bundle,
                product,
                category,
                previous_price,
                target_date.strftime("%Y-%m-%d"),
                previous_demand,
            )
            change = value - previous_price
            percentage = (change / previous_price) * 100

            c1, c2, c3 = st.columns(3)
            c1.metric(
                "Estimasi harga",
                rupiah(value),
                f"{change:+,.0f}",
            )
            c2.metric(
                "Perubahan",
                f"{percentage:+.2f}%",
            )
            c3.metric(
                "Periode",
                target_date.strftime("%B %Y"),
            )

            st.caption(
                "Hasil adalah estimasi berdasarkan pola historis dataset, "
                "bukan jaminan harga pasar."
            )

    else:
        st.subheader("Perbandingan model")

        if not bundle:
            st.info("Latih model terlebih dahulu.")
            return

        st.dataframe(
            bundle["metrics"],
            use_container_width=True,
            hide_index=True,
        )
        st.caption(
            f"Model utama: {RF_NAME}. Pembanding: {LR_NAME} dan "
            f"{BASELINE_NAME}."
        )
        st.bar_chart(
            bundle["metrics"]
            .set_index("Model")[["MAE", "RMSE"]]
        )


if __name__ == "__main__":
    main()
