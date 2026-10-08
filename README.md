# AI Price Predictor

**AI Price Predictor** adalah aplikasi **Python + Streamlit + Scikit-learn** untuk memperkirakan **harga produk pada periode berikutnya berdasarkan data historis**. Aplikasi mendukung dataset **CSV, XLSX, dan XLS**.

Project ini dibuat untuk mata kuliah **Kecerdasan Buatan** dan berjalan secara lokal tanpa API AI eksternal.

## 1. Konsep Sistem

Pertanyaan utama yang dijawab sistem:

> **Berdasarkan riwayat suatu produk, berapa estimasi harganya pada periode berikutnya?**

Target model:

```text
price = harga aktual produk pada periode target
```

Fitur model:

```text
year
month
day
previous_price
previous_demand
product
category
```

`previous_price` dan `previous_demand` berasal dari periode sebelumnya sehingga sistem tidak meminta informasi masa depan.

## 2. Alur AI

```text
Dataset historis
      ↓
Preprocessing
      ↓
Feature Engineering
      ↓
Pembagian train/test berdasarkan waktu
      ↓
Training Random Forest
      ↓
Evaluasi
      ↓
Model terlatih
      ↓
Data terakhir produk
      ↓
Periode berikutnya
      ↓
Estimasi harga
```

## 3. Fitur Aplikasi

- Dashboard dan grafik tren harga
- Upload dataset CSV / XLSX / XLS
- Pemetaan kolom otomatis
- Deteksi format tanggal dan angka
- Validasi dan preprocessing
- Training Random Forest Regressor
- Perbandingan Linear Regression dan baseline
- Evaluasi MAE, MSE, RMSE, dan R²
- Prediksi otomatis periode berikutnya
- Perhitungan perubahan harga dan persentase

## 4. Teknologi

- Python 3.12
- Streamlit
- Pandas
- NumPy
- Scikit-learn
- Joblib
- Matplotlib
- openpyxl
- xlrd

## 5. Struktur Project

```text
ai-predictor/
├── app.py
├── requirements.txt
├── run.bat
├── README.md
├── PRESENTASI.md
├── src/
│   ├── preprocessing.py
│   ├── import_data.py
│   ├── auto_import.py
│   ├── train.py
│   ├── predict.py
│   ├── generate_data.py
│   └── __init__.py
├── data/
└── models/
```

## 6. Instalasi dan Menjalankan Aplikasi

### 6.1 Clone repository

```powershell
git clone https://github.com/Hrsxr9/ai-predictor.git
cd ai-predictor
```

### 6.2 Cek Python

```powershell
py -3.12 --version
```

### 6.3 Buat virtual environment

```powershell
py -3.12 -m venv .venv
```

### 6.4 Install dependency

Normal:

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Jika `Activate.ps1` diblokir PowerShell, langsung gunakan Python dari `.venv`:

```powershell
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

### 6.5 Jalankan aplikasi

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Atau setelah venv aktif:

```powershell
python -m streamlit run app.py
```

Buka alamat yang ditampilkan Streamlit, biasanya `http://localhost:8501`.

### 6.6 Cara cepat

Windows juga dapat menjalankan:

```text
run.bat
```

## 7. Menggunakan Dataset Contoh

Ini adalah alur demo paling cepat.

```text
1. Jalankan aplikasi
2. Pilih sumber: Dataset contoh
3. Buka menu Training Model
4. Klik Latih model
5. Tunggu training selesai
6. Buka Prediksi Periode Berikutnya
7. Pilih produk
8. Klik Prediksi Harga Berikutnya
9. Lihat estimasi harga
```

Dataset contoh dibuat otomatis apabila belum tersedia.

## Smart Mapping

Saat file di-upload, aplikasi tidak hanya mencocokkan nama kolom. **Smart Mapping** menganalisis nama kolom, tipe data, pola isi, dan hubungan antar-kolom untuk menentukan peran data.

Hasilnya ditampilkan pada menu **Dataset → Smart Mapping**, termasuk:

- peran yang ditemukan (tanggal, produk, harga, demand, dan lainnya)
- nama kolom asli
- tingkat keyakinan
- alasan pemetaan

Untuk dataset transaksi/e-commerce, aplikasi juga dapat mengenali pola seperti:

```text
Waktu Pesanan Dibuat → tanggal
product_categories   → kelompok produk
Total Pembayaran     → nilai harga/transaksi
total_qty            → jumlah/demand
```

Jika tersedia `Total Pembayaran` dan `total_qty`, sistem dapat membentuk **proxy harga per unit = Total Pembayaran ÷ total_qty**, lalu mengagregasikan transaksi per produk dan tanggal sebelum training. Ini adalah transformasi data, bukan klaim bahwa kolom tersebut selalu merupakan harga satuan asli.

## 8. Upload Dataset Sendiri

Aplikasi menerima:

```text
.csv
.xlsx
.xls
```

### 8.1 CSV

Delimiter yang dicoba otomatis:

```text
,   ;   tab   |
```

### 8.2 Excel

Workbook boleh memiliki beberapa sheet. Aplikasi memilih sheet yang paling layak berdasarkan struktur dan jumlah data.

### 8.3 Kolom minimum

Dataset wajib mempunyai:

- tanggal
- produk
- harga aktual

Opsional:

- kategori
- previous_price
- demand / qty / quantity / jumlah_terjual

Jika `previous_price` tidak tersedia, sistem menurunkannya dari histori harga per produk.

Jika demand tersedia, sistem membentuk `previous_demand` dari periode sebelumnya. Untuk dataset transaksi/e-commerce, sistem dapat mengenali kolom seperti `Waktu Pesanan Dibuat`, `product_categories`, `Total Pembayaran`, dan `total_qty`, lalu mengubah transaksi menjadi observasi harga yang dapat dipakai model. Untuk pola transaksi ini, harga yang dipakai adalah **estimasi harga per unit = Total Pembayaran ÷ total_qty**, kemudian diagregasi per produk dan tanggal.

### 8.4 Contoh dataset

```csv
date,product,category,price,demand
2025-01-01,Beras,Sembako,13150,820
2025-02-01,Beras,Sembako,13300,850
2025-03-01,Beras,Sembako,13450,870
```

Nama kolom tidak harus persis sama. **Smart Mapping** menganalisis nama kolom, tipe data, dan pola isi untuk menentukan peran seperti tanggal, produk, harga, dan demand. Hasil pemetaan dapat diperiksa pada menu **Dataset → Smart Mapping** beserta tingkat keyakinan dan alasannya.

## 9. Urutan Menggunakan Dataset Sendiri

Setelah upload file, ikuti urutan ini:

```text
1. Pilih Upload Dataset
        ↓
2. Upload CSV / XLSX / XLS
        ↓
3. Sistem membaca dan memetakan kolom
        ↓
4. Buka menu Dataset
        ↓
5. Periksa data hasil preprocessing
        ↓
6. Buka Training Model
        ↓
7. Klik Latih model
        ↓
8. Lihat hasil evaluasi
        ↓
9. Buka Prediksi Periode Berikutnya
        ↓
10. Pilih produk
        ↓
11. Klik Prediksi Harga Berikutnya
        ↓
12. Lihat estimasi harga periode berikutnya
```

**Setiap kali dataset diganti, lakukan training ulang.**

## 10. Contoh Prediksi

Misalnya data terakhir produk:

```text
Produk           : Beras
Periode terakhir : September 2026
Harga terakhir   : Rp15.000
Demand terakhir  : 950
```

Sistem otomatis menentukan target:

```text
Periode target   : Oktober 2026
previous_price   : Rp15.000
previous_demand  : 950
```

Random Forest kemudian menghasilkan estimasi harga Oktober 2026.

Output aplikasi:

```text
Estimasi harga
Perubahan nominal
Perubahan persentase
Periode prediksi
```

Pengguna **tidak perlu memasukkan demand atau harga masa depan**.

## 11. Training dan Evaluasi

Model utama:

```text
Random Forest Regressor
```

Model pembanding:

```text
Linear Regression
Baseline = harga periode sebelumnya
```

Pembagian data menggunakan **urutan waktu**, bukan random. Periode lama digunakan untuk training dan periode lebih baru digunakan untuk testing.

Metode evaluasi:

- MAE
- MSE
- RMSE
- R²

R² bukan accuracy classification.

## 12. Dataset Demo

Dataset bawaan merupakan data sintetis untuk demonstrasi:

```text
480 baris
6 produk
80 periode bulanan per produk
Januari 2020 – Agustus 2026
```

Dataset sintetis bukan data harga pasar nyata, sehingga hasil evaluasinya tidak boleh dianggap sebagai performa pada seluruh pasar.

## 13. Troubleshooting

### Python tidak ditemukan

```powershell
py -3.12 --version
```

### PowerShell menolak Activate.ps1

Gunakan:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

### Dependency belum terpasang

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

### Model tidak tersedia atau dataset berubah

Buka:

```text
Training Model → Latih model
```

## 14. AI-nya Di Mana?

AI/ML berada pada proses:

```text
Training Random Forest Regressor
             ↓
     Model belajar pola
             ↓
      Prediksi harga
```

Streamlit hanya digunakan sebagai antarmuka aplikasi.

## 15. Batasan Sistem

- Dataset demo masih sintetis.
- Hasil prediksi bergantung pada kualitas data historis.
- Faktor eksternal seperti inflasi, promo, cuaca, distribusi, dan kebijakan belum digunakan.
- Prediksi merupakan estimasi model, bukan jaminan harga pasar.
- Sistem berfokus pada satu periode berikutnya.

## 16. Kesimpulan

AI Price Predictor menerapkan **supervised machine learning pada masalah regression** untuk memperkirakan **harga produk pada periode berikutnya berdasarkan data historis**.

```text
Dataset
→ Preprocessing
→ Feature Engineering
→ Training
→ Evaluasi
→ Model
→ Data Terbaru
→ Prediksi Periode Berikutnya
```

Model utama adalah **Random Forest Regressor**, dengan **Linear Regression** dan **baseline harga sebelumnya** sebagai pembanding.