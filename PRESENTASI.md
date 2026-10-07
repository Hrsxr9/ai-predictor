# PRESENTASI — AI Price Predictor

## 1. Identitas Project

**Judul:** AI Price Predictor: Sistem Prediksi Harga Barang Berdasarkan Data Historis  
**Mata Kuliah:** Kecerdasan Buatan  
**Platform:** Web lokal menggunakan Streamlit  
**Bahasa:** Python  
**Pendekatan:** Supervised Machine Learning — Regression  
**Algoritma utama:** Random Forest Regressor  
**Model pembanding:** Linear Regression dan baseline harga sebelumnya

Project berjalan secara lokal dan tidak membutuhkan API AI eksternal.

## 2. Latar Belakang

Harga suatu barang dapat berubah dari waktu ke waktu. Perubahan tersebut dapat terlihat dari riwayat harga produk dan informasi pendukung seperti demand.

Project ini menerapkan machine learning untuk menjawab satu pertanyaan utama:

> **Berdasarkan data historis suatu produk, berapa estimasi harga pada periode berikutnya?**

Sistem mempelajari hubungan antara informasi yang sudah diketahui sampai periode terakhir dengan harga aktual pada periode berikutnya.

## 3. Tujuan

1. Membuat sistem prediksi harga berbasis machine learning.
2. Menerapkan supervised learning pada permasalahan regression.
3. Membandingkan Random Forest dengan Linear Regression dan baseline sederhana.
4. Menampilkan evaluasi dan hasil prediksi melalui aplikasi web.
5. Membuat alur prediksi yang tidak membutuhkan data masa depan dari pengguna.

## 4. Konsep AI

**Input dari data historis:**

- Produk
- Kategori
- Tanggal
- Harga periode sebelumnya
- Demand periode sebelumnya

**Target:**

- Harga aktual pada periode yang diprediksi

Alurnya:

```text
Data historis
      ↓
Preprocessing
      ↓
Feature Engineering
      ↓
Training
      ↓
Random Forest Regressor
      ↓
Model terlatih
      ↓
Data terbaru suatu produk
      ↓
Target periode berikutnya
      ↓
Estimasi harga
```

## 5. Mengapa Regression?

Output yang ingin diprediksi adalah **harga**, yaitu nilai numerik kontinu.

Contoh:

```text
Rp14.850
Rp15.120
Rp15.430
```

Karena target berupa angka kontinu, masalah ini termasuk **regression**, bukan classification.

## 6. Fitur yang Digunakan

Fitur numerik:

```text
year
month
day
previous_price
previous_demand
```

Fitur kategorikal:

```text
product
category
```

Target:

```text
price
```

### Kenapa menggunakan previous_demand?

Karena ketika memprediksi periode berikutnya, demand pada periode target belum diketahui. Oleh sebab itu sistem menggunakan **demand pada periode sebelumnya**, bukan demand masa depan.

## 7. Preprocessing

```text
CSV
 ↓
Validasi kolom
 ↓
Validasi tanggal
 ↓
Validasi produk
 ↓
Validasi harga
 ↓
Penanganan duplikat
 ↓
Penanganan missing value
 ↓
Membentuk previous_price
 ↓
Membentuk previous_demand
 ↓
Feature engineering tanggal
 ↓
Data siap training
```

Sistem juga mempunyai pemetaan kolom otomatis sehingga nama seperti `tanggal`, `nama_barang`, dan `harga` dapat dipetakan ke schema internal.

## 8. Pembagian Data

Data tidak dibagi secara acak.

```text
Periode lama                          Periode baru
|----------------------------------------------|
          TRAIN                 TEST
            80%                  20%
```

Tujuannya adalah mensimulasikan kondisi nyata:

> model belajar dari masa lalu dan diuji pada periode yang lebih baru.

## 9. Model Machine Learning

### Random Forest Regressor

Random Forest terdiri dari banyak decision tree. Prediksi dari beberapa tree digabungkan untuk menghasilkan prediksi akhir.

Konfigurasi utama:

```text
n_estimators = 160
max_depth = 12
min_samples_leaf = 2
random_state = 42
```

### Model pembanding

**Linear Regression** digunakan sebagai model pembanding sederhana.

**Baseline** menggunakan harga periode sebelumnya sebagai estimasi.

## 10. Evaluasi

**MAE**  
Rata-rata besar kesalahan absolut prediksi.

**MSE**  
Rata-rata kuadrat kesalahan.

**RMSE**  
Akar dari MSE sehingga kembali ke satuan harga.

**R²**  
Mengukur kemampuan model menjelaskan variasi target pada data pengujian.

> R² bukan accuracy classification.

## 11. Dataset Demonstrasi

Project menyediakan dataset sintetis untuk demonstrasi pipeline:

- 480 baris
- 6 produk
- 80 periode bulanan per produk
- Januari 2020 sampai Agustus 2026

Data sintetis dipakai agar pipeline dapat didemonstrasikan tanpa bergantung pada data eksternal.

## 12. Contoh Skenario Prediksi

Misalnya data terakhir:

```text
Produk           : Beras Medium
Data terakhir    : Juli 2026
Harga terakhir   : Rp15.000
Demand terakhir  : 950
```

Sistem otomatis menentukan:

```text
Periode target   : Agustus 2026
previous_price   : Rp15.000
previous_demand  : 950
```

Random Forest kemudian menghasilkan:

```text
Estimasi harga Agustus 2026
= hasil prediksi model
```

Pengguna tidak memasukkan harga atau demand masa depan.

## 13. Output Aplikasi

Aplikasi menampilkan:

- Harga terakhir
- Periode yang diprediksi
- Estimasi harga
- Nilai perubahan harga
- Persentase perubahan

Contoh tampilan konsep:

```text
Harga terakhir       Rp15.000
↓
Prediksi bulan depan Rp15.240
↓
Perubahan            +Rp240
Persentase           +1,60%
```

## 14. Arsitektur Sistem

```text
┌───────────────────────────────┐
│          Streamlit            │
│        User Interface         │
└───────────────┬───────────────┘
                │
                ▼
┌───────────────────────────────┐
│       Application Logic       │
│ Python / Pandas / NumPy       │
└───────────────┬───────────────┘
                │
       ┌────────┴─────────┐
       ▼                  ▼
Preprocessing         Machine Learning
       │                  │
       └────────┬─────────┘
                ▼
       Random Forest / LR
                │
                ▼
          Model / Predict
```

## 15. Struktur Program

```text
app.py                  → tampilan dan alur aplikasi
src/preprocessing.py    → validasi, cleaning, feature engineering
src/import_data.py      → mapping dan normalisasi CSV
src/auto_import.py      → deteksi format tanggal dan angka
src/train.py            → training dan evaluasi
src/predict.py          → prediksi periode berikutnya
src/generate_data.py    → pembuatan dataset demo sintetis
```

## 16. Alur Penggunaan Aplikasi

**Dashboard**  
Melihat ringkasan dataset dan tren harga.

**Dataset**  
Melihat dataset aktif dan hasil validasi.

**Training Model**  
Melatih model dan melihat evaluasi.

**Prediksi Periode Berikutnya**  
Pilih produk, lihat periode target otomatis, lalu jalankan prediksi.

**Model Comparison**  
Membandingkan Random Forest, Linear Regression, dan baseline.

## 17. Demo Singkat

```text
1. Jalankan aplikasi
2. Buka Training Model
3. Klik "Latih model"
4. Lihat MAE, RMSE, dan R²
5. Buka "Prediksi Periode Berikutnya"
6. Pilih produk
7. Lihat harga terakhir dan target bulan berikutnya
8. Klik "Prediksi Harga Berikutnya"
9. Tampilkan estimasi harga dan persentase perubahan
```

## 18. Jawaban Saat Ditanya Dosen

**AI-nya di mana?**  
AI/ML berada pada proses training dan prediksi menggunakan Random Forest Regressor. Streamlit hanya menjadi antarmuka.

**Apa yang sebenarnya diprediksi?**  
Estimasi harga produk pada periode berikutnya berdasarkan pola data historis.

**Kenapa regression?**  
Karena target yang diprediksi adalah nilai harga numerik kontinu.

**Kenapa menggunakan previous_price?**  
Karena harga periode sebelumnya merupakan informasi yang sudah diketahui dan berhubungan langsung dengan kondisi harga saat ini.

**Kenapa previous_demand, bukan demand?**  
Karena demand pada periode target belum diketahui. Menggunakan demand sebelumnya menghindari kebutuhan memasukkan informasi masa depan.

**Kenapa data dibagi berdasarkan waktu?**  
Agar pengujian menyerupai kondisi nyata: belajar dari masa lalu dan menguji pada periode yang lebih baru.

**Apa baseline?**  
Harga periode sebelumnya digunakan sebagai prediksi sederhana tanpa machine learning.

**Apakah R² 0,9949 berarti akurasi 99,49%?**  
Tidak. R² bukan accuracy classification.

**Apakah data nyata?**  
Dataset demo adalah sintetis. Untuk penggunaan nyata diperlukan data historis riil.

**Apakah prediksi menjamin harga pasar?**  
Tidak. Output adalah estimasi berdasarkan pola dataset.

## 19. Keterbatasan

1. Dataset demo masih sintetis.
2. Faktor eksternal seperti inflasi, promo, cuaca, distribusi, dan kebijakan belum dimasukkan.
3. Kualitas prediksi sangat bergantung pada kualitas data historis.
4. Hasil pada dataset demo tidak otomatis mewakili performa di dunia nyata.
5. Model belum dibuat sebagai forecasting multi-langkah untuk banyak periode ke depan.

## 20. Pengembangan Selanjutnya

- Menggunakan dataset harga riil.
- Menambahkan faktor eksternal yang relevan.
- Hyperparameter tuning.
- Membandingkan algoritma tambahan.
- Menampilkan grafik aktual vs prediksi.
- Menyimpan histori prediksi.
- Menambahkan interval ketidakpastian.
- Mengembangkan forecasting beberapa periode sekaligus.

## 21. Kesimpulan

AI Price Predictor menerapkan **supervised machine learning — regression** untuk memperkirakan **harga produk pada periode berikutnya** berdasarkan informasi historis yang tersedia sampai periode terakhir.

```text
Data Historis
→ Preprocessing
→ Feature Engineering
→ Training
→ Evaluasi
→ Model
→ Data Terbaru
→ Prediksi Periode Berikutnya
```

Model utama adalah **Random Forest Regressor**, dengan Linear Regression dan baseline harga sebelumnya sebagai pembanding.

Project dapat dijalankan secara lokal menggunakan Python dan Streamlit tanpa API AI eksternal.
