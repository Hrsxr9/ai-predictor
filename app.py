"""Streamlit UI untuk AI Price Predictor."""

from pathlib import Path

import pandas as pd
import streamlit as st

from src.generate_data import generate_dataset
from src.import_data import import_dataset, suggest_mapping, DATE_FORMATS, NUMBER_FORMATS
from src.preprocessing import DEFAULT_DATA, clean_data, dataset_fingerprint, rupiah
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
        generate_dataset().to_csv(DEFAULT_DATA, index=False, date_format="%Y-%m-%d")


def load_active_data(uploaded=None):
    if uploaded is None:
        raw = pd.read_csv(DEFAULT_DATA)
        frame, report = clean_data(raw)
        return frame, report, "Dataset contoh"
    raw = pd.read_csv(uploaded)
    mapping = suggest_mapping(raw.columns)
    st.sidebar.caption("Pemetaan otomatis: periksa sebelum digunakan.")
    with st.sidebar.form("mapping_form"):
        selected = {}
        labels = {
            "date": "Tanggal",
            "product": "Produk",
            "category": "Kategori",
            "previous_price": "Harga sebelumnya",
            "demand": "Demand",
            "price": "Harga aktual",
        }
        options = [None] + list(raw.columns)
        for key, label in labels.items():
            current = mapping.get(key)
            selected[key] = st.selectbox(
                label,
                options,
                index=options.index(current) if current in options else 0,
                format_func=lambda x: "— Tidak dipakai —" if x is None else str(x),
                key=f"map_{key}",
            )
        submitted = st.form_submit_button("Gunakan CSV")
    if not submitted:
        st.dataframe(raw.head(10), use_container_width=True, hide_index=True)
        st.info("Pilih kolom di sidebar lalu klik Gunakan CSV.")
        return None
    frame, report = import_dataset(raw, selected)
    return frame, report, uploaded.name


def main():
    ensure_demo_data()

    st.title("AI Price Predictor")
    st.caption("Sistem Prediksi Harga Barang Berdasarkan Data Historis · Full Local")

    with st.sidebar:
        st.header("Dataset")
        source = st.radio("Sumber", ["Dataset contoh", "Upload CSV"])
        uploaded = st.file_uploader("CSV", type=["csv"], disabled=source != "Upload CSV")
        st.divider()
        page = st.radio(
            "Menu",
            ["Dashboard", "Dataset", "Training Model", "Prediksi Harga", "Model Comparison"],
        )

    uploaded_obj = uploaded if source == "Upload CSV" else None
    try:
        active = load_active_data(uploaded_obj)
    except (OSError, ValueError) as exc:
        st.error(str(exc))
        return
    if active is None:
        return

    frame, report, source_name = active
    fingerprint = dataset_fingerprint(frame)

    if st.session_state.get("dataset_hash") != fingerprint:
        st.session_state["dataset_hash"] = fingerprint
        st.session_state.pop("bundle", None)
        try:
            path = Path("models/model.pkl")
            if path.exists():
                st.session_state["bundle"] = load_model(path, fingerprint)
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
        product = st.selectbox("Produk", sorted(frame["product"].unique()))
        history = frame[frame["product"] == product].set_index("date")[["price"]]
        st.line_chart(history.rename(columns={"price": "Harga (Rp)"}))

        st.info(
            "Alur AI: data historis → preprocessing → feature engineering → "
            "training → evaluasi → prediksi."
        )

    elif page == "Dataset":
        st.subheader("Dataset aktif")
        st.write(f"Sumber: **{source_name}**")
        st.dataframe(frame, use_container_width=True, hide_index=True)
        st.write(
            {
                "Baris masuk": report.get("rows_in"),
                "Baris valid": report.get("rows_out"),
                "Baris dibuang": report.get("rows_removed"),
                "Previous price kosong": report.get("previous_price_missing"),
                "Demand kosong": report.get("demand_missing"),
            }
        )

    elif page == "Training Model":
        st.subheader("Training & evaluasi")
        st.write(
            "Data dibagi berdasarkan urutan waktu. Random Forest dan Linear Regression "
            "dibandingkan dengan baseline harga sebelumnya."
        )
        if st.button("Latih model", type="primary"):
            with st.spinner("Training..."):
                bundle = train_models(frame)
                Path("models").mkdir(exist_ok=True)
                save_model(bundle, Path("models/model.pkl"))
                st.session_state["bundle"] = bundle
            st.success("Model selesai dilatih dan Random Forest disimpan untuk prediksi.")

        if bundle:
            st.dataframe(bundle["metrics"], use_container_width=True, hide_index=True)
            st.caption(
                f"Train {bundle['split']['train_rows']} baris · "
                f"Test {bundle['split']['test_rows']} baris"
            )

    elif page == "Prediksi Harga":
        st.subheader("Prediksi harga")
        if not bundle:
            st.info("Latih model terlebih dahulu.")
            return

        product = st.selectbox("Produk", sorted(frame["product"].unique()))
        category = frame.loc[frame["product"] == product, "category"].mode().iat[0]
        target_date = st.date_input("Tanggal target", value=pd.Timestamp.now().date())
        previous_price = st.number_input(
            "Harga sebelumnya (Rp)",
            min_value=1.0,
            value=float(frame.loc[frame["product"] == product, "price"].iloc[-1]),
            step=100.0,
        )
        demand_values = frame.loc[frame["product"] == product, "demand"].dropna()
        demand = st.number_input(
            "Demand",
            min_value=0.0,
            value=float(demand_values.iloc[-1]) if not demand_values.empty else 0.0,
            step=1.0,
        )

        if st.button("Prediksi", type="primary"):
            value = predict_price(
                bundle,
                product,
                category,
                previous_price,
                target_date.strftime("%Y-%m-%d"),
                demand,
            )
            st.metric("Estimasi harga", rupiah(value))
            st.caption(
                "Hasil adalah estimasi berdasarkan pola dataset, bukan jaminan harga pasar."
            )

    else:
        st.subheader("Perbandingan model")
        if not bundle:
            st.info("Latih model terlebih dahulu.")
            return
        st.dataframe(bundle["metrics"], use_container_width=True, hide_index=True)
        st.caption(
            f"Model utama: {RF_NAME}. Pembanding: {LR_NAME} dan {BASELINE_NAME}. "
            "Kesimpulan berlaku untuk dataset dan pembagian waktu ini."
        )
        st.bar_chart(bundle["metrics"].set_index("Model")[["MAE", "RMSE"]])


if __name__ == "__main__":
    main()
