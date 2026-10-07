# AI Price Predictor

**AI Price Predictor** adalah aplikasi berbasis **Python + Streamlit + Scikit-learn** untuk memperkirakan **harga produk pada periode berikutnya** berdasarkan pola data historis.

Project ini dibuat untuk mata kuliah **Kecerdasan Buatan** dan berjalan **full local**, tanpa OpenAI API, Gemini API, OpenRouter, atau layanan AI eksternal.

## Konsep Utama

Pertanyaan yang dijawab sistem:

> **"Berdasarkan riwayat harga suatu produk, berapa estimasi harga pada periode berikutnya?"**

Target machine learning:

`price` = harga aktual produk pada periode yang dipelajari.

Fitur yang digunakan:

- `year`, `month`, `day`
- `previous_price` = harga produk pada periode sebelumnya
- `previous_demand` = demand pada periode sebelumnya
- `product`
- `category`

Sistem **tidak meminta demand masa depan**. Demand yang digunakan sebagai fitur berasal dari periode sebelumnya. Bila data demand tidak tersedia, fitur tersebut dapat diisi otomatis dan akan ditangani saat training.

## Fitur

- Membaca dataset harga historis dari CSV
- Validasi dan preprocessing data
- Feature engineering dari tanggal
- Membuat fitur historis `previous_price` dan `previous_demand`
- Perbandingan Random Forest Regressor, Linear Regression, dan baseline harga sebelumnya
- Evaluasi dengan MAE, MSE, RMSE, dan R²
- **Prediksi otomatis periode berikutnya**
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
│   ├── predict.py          # Prediksi periode berikutnya
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

### 3. Buat virtual environment

```powershell
py -3.12 -m venv .venv
```

### 4. Aktifkan virtual environment

PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Kalau PowerShell menolak script, tidak perlu mengubah execution policy. Gunakan executable Python dari folder `.venv` secara langsung.

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
8. Buka menu "Prediksi Periode Berikutnya"
        ↓
9. Pilih produk
        ↓
10. Klik "Prediksi Harga Berikutnya"
        ↓
11. Tampilkan estimasi harga periode berikutnya
```

## Skenario Prediksi

Misalnya data terakhir suatu produk adalah:

```text
Produk          : Beras Medium
Periode terakhir: Juli 2026
Harga terakhir  : Rp15.000
Demand terakhir : 950
```

Saat tombol prediksi dijalankan, sistem otomatis membuat:

```text
Target periode: Agustus 2026
previous_price = Rp15.000
previous_demand = 950
```

Model kemudian menghasilkan:

```text
Estimasi harga Agustus 2026 = hasil prediksi Random Forest
```

Jadi pengguna **tidak perlu menebak atau mengisi data masa depan**.

## Menggunakan Dataset Sendiri

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

Jika `previous_price` tidak tersedia, sistem dapat menurunkannya otomatis dari harga historis. Jika demand tersedia, sistem juga otomatis membuat `previous_demand` berdasarkan periode sebelumnya.

## Konsep Machine Learning

### Target

```text
price
```

### Fitur

```text
year
month
day
previous_price
previous_demand
product
category
```

### Algoritma utama

```text
Random Forest Regressor
```

### Model pembanding

```text
Linear Regression
Baseline = harga periode sebelumnya
```

Pembagian data menggunakan **urutan waktu**, sehingga periode yang lebih baru menjadi data pengujian. Ini mensimulasikan skenario belajar dari masa lalu lalu mengestimasi periode berikutnya.

## Evaluasi

Model dievaluasi menggunakan:

- **MAE** — rata-rata kesalahan absolut
- **MSE** — rata-rata kuadrat kesalahan
- **RMSE** — akar dari MSE dan masih dalam satuan harga
- **R²** — kemampuan model menjelaskan variasi target pada data pengujian

Nilai evaluasi hanya berlaku untuk dataset dan pembagian train/test pada eksperimen tersebut.

## Catatan Dataset Demo

Dataset bawaan adalah **data sintetis untuk demonstrasi pipeline machine learning**, bukan data harga pasar aktual.

Karena itu:

- hasil evaluasi tidak boleh dianggap sebagai performa pada seluruh pasar;
- hasil prediksi tidak boleh dianggap sebagai harga pasar yang pasti.

Untuk penggunaan nyata, gunakan data historis riil dan fitur yang relevan dengan produk yang diprediksi.

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

### Package belum terpasang

```powershell
python -m pip install -r requirements.txt
```

### Dataset berubah dan model lama tidak bisa dipakai

Itu memang dibuat sebagai pengaman. Setelah dataset berubah, jalankan:

**Training Model → Latih model**

## Penggunaan Akademik

Project ini dibuat untuk pembelajaran dan demonstrasi mata kuliah Kecerdasan Buatan.
