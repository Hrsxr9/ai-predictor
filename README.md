# AI Price Predictor

**AI Price Predictor** adalah aplikasi berbasis **Python + Streamlit + Scikit-learn** untuk memperkirakan harga barang berdasarkan data historis.

Project ini dibuat untuk mata kuliah **Kecerdasan Buatan** dan berjalan **full local**, tanpa OpenAI API, Gemini API, OpenRouter, atau layanan AI eksternal.

## Fitur

- Membaca dataset harga historis dari CSV
- Validasi dan preprocessing data
- Feature engineering dari tanggal
- Penanganan missing value saat training
- Perbandingan Random Forest Regressor, Linear Regression, dan baseline harga sebelumnya
- Evaluasi dengan MAE, MSE, RMSE, dan R²
- Prediksi harga dari input baru
- Visualisasi tren harga
- Import CSV dengan pemetaan kolom
- Dataset demonstrasi sintetis yang dibuat lokal

## Teknologi

- Python 3.12
- Streamlit
- Pandas
- NumPy
- Scikit-learn
- Joblib
- Matplotlib

## Struktur Project

```text
ai-predictor/
├── app.py                  # Entry point aplikasi Streamlit
├── requirements.txt        # Dependency Python
├── run.bat                 # Shortcut Windows untuk menjalankan aplikasi
├── README.md
├── PRESENTASI.md           # Materi penjelasan/presentasi project
├── src/
│   ├── preprocessing.py    # Validasi, cleaning, feature engineering
│   ├── import_data.py      # Import dan mapping CSV
│   ├── auto_import.py      # Deteksi format tanggal/angka
│   ├── train.py            # Training dan evaluasi model
│   ├── predict.py          # Prediksi menggunakan model
│   ├── generate_data.py    # Generator dataset demo sintetis
│   └── __init__.py
├── data/                   # Dataset demo dibuat otomatis saat diperlukan
└── models/                 # Model terlatih disimpan lokal
```

## Cara Menjalankan di Windows

### 1. Clone repository

Buka PowerShell:

```powershell
git clone https://github.com/Hrsxr9/ai-predictor.git
cd ai-predictor
```

### 2. Cek Python

Project ini ditujukan untuk Python 3.12.

```powershell
py -3.12 --version
```

Hasil yang diharapkan kurang lebih:

```text
Python 3.12.x
```

### 3. Buat virtual environment

```powershell
py -3.12 -m venv .venv
```

### 4. Aktifkan virtual environment

PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Kalau PowerShell menolak script, tidak perlu mengubah execution policy. Gunakan executable Python dari folder `.venv` langsung seperti pada langkah berikutnya.

### 5. Install dependency

Jika venv sudah aktif:

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Tanpa aktivasi:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

### 6. Jalankan aplikasi

**Perintah utama yang harus dijalankan:**

```powershell
python -m streamlit run app.py
```

Tanpa mengaktifkan venv:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Biasanya aplikasi dapat dibuka di:

```text
http://localhost:8501
```

### Cara paling gampang di Windows

Setelah repository selesai di-clone, jalankan:

```text
run.bat
```

Script akan membuat `.venv` bila belum ada, memasang dependency, lalu menjalankan Streamlit.

## Urutan Demo dari Nol

**Tidak perlu menjalankan file Python satu per satu sebelum membuka aplikasi.**

Urutan yang benar:

```text
1. Clone repository
        ↓
2. Buat .venv
        ↓
3. Install requirements.txt
        ↓
4. Jalankan app.py dengan Streamlit
        ↓
5. Dataset contoh otomatis dibuat bila belum ada
        ↓
6. Buka menu "Training Model"
        ↓
7. Klik "Latih model"
        ↓
8. Buka "Prediksi Harga"
```

Saat aplikasi pertama kali berjalan, file berikut akan dibuat lokal bila belum ada:

```text
data/historical_prices.csv
```

Setelah training berhasil, model disimpan sebagai:

```text
models/model.pkl
```

File model tidak di-commit ke GitHub karena masuk `.gitignore`.

## Menggunakan Dataset Sendiri

Di aplikasi pilih:

**Dataset → Upload CSV**

Minimal dataset harus memiliki:

- tanggal
- produk
- harga aktual

Kolom kategori, harga sebelumnya, dan demand dapat digunakan jika tersedia.

Contoh:

```csv
date,product,category,previous_price,demand,price
2025-01-01,Beras,Sembako,13000,820,13150
2025-02-01,Beras,Sembako,13150,850,13300
2025-03-01,Beras,Sembako,13300,870,13450
```

Setelah CSV dipilih, periksa pemetaan kolom di sidebar lalu klik **Gunakan CSV**.

## Konsep Machine Learning

Target:

```text
price
```

Fitur utama:

```text
year
month
day
previous_price
demand
product
category
```

Algoritma utama:

```text
Random Forest Regressor
```

Model pembanding:

```text
Linear Regression
Baseline = harga sebelumnya
```

Pembagian data menggunakan **urutan waktu**, sehingga periode yang lebih baru menjadi data pengujian.

## Evaluasi

Model dievaluasi menggunakan:

- **MAE** — rata-rata kesalahan absolut
- **MSE** — rata-rata kuadrat kesalahan
- **RMSE** — akar dari MSE
- **R²** — kemampuan model menjelaskan variasi target pada data pengujian

Nilai evaluasi hanya berlaku untuk dataset dan pembagian train/test pada eksperimen tersebut.

## Catatan Dataset Demo

Dataset bawaan adalah **data sintetis untuk demonstrasi pipeline machine learning**, bukan data harga pasar aktual.

Karena itu hasil prediksi dari dataset demo tidak boleh dianggap sebagai harga pasar nyata.

## Troubleshooting

### `python` tidak ditemukan

Gunakan:

```powershell
py -3.12 -m streamlit run app.py
```

atau:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

### `streamlit` tidak ditemukan

Gunakan:

```powershell
python -m streamlit run app.py
```

jangan hanya:

```powershell
streamlit run app.py
```

### Package belum terpasang

```powershell
python -m pip install -r requirements.txt
```

### Dataset berubah dan model lama tidak bisa dipakai

Itu memang dibuat sebagai pengaman. Setelah dataset berubah, jalankan:

**Training Model → Latih model**

## Penggunaan Akademik

Project ini dibuat untuk pembelajaran dan demonstrasi mata kuliah Kecerdasan Buatan.
