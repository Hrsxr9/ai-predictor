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
Harga suatu barang dapat berubah dari waktu ke waktu. Perubahan tersebut dapat dipengaruhi oleh kondisi historis seperti harga periode sebelumnya, tingkat permintaan, jenis produk, kategori, dan periode waktu.
Project ini dibuat untuk menerapkan machine learning pada masalah prediksi harga. Sistem mempelajari pola dari data historis, kemudian menggunakan pola tersebut untuk menghasilkan estimasi harga pada data baru.

## 3. Tujuan
1. Membuat sistem prediksi harga berbasis machine learning.
2. Menerapkan supervised learning pada permasalahan regression.
3. Membandingkan beberapa pendekatan model.
4. Menampilkan hasil prediksi dan evaluasi dalam aplikasi web lokal.
5. Membuat project yang dapat digunakan dengan dataset historis milik pengguna.

## 4. Konsep AI
Input sistem: Produk, Kategori, Tanggal, Harga periode sebelumnya, dan Demand.
Target: Harga barang pada periode yang diprediksi.

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
Model
      ↓
Input baru
      ↓
Estimasi harga
```

## 5. Mengapa Regression?
Output yang ingin diprediksi berupa angka kontinu, yaitu harga. Karena target berupa nilai numerik kontinu, masalah ini termasuk regression.

## 6. Dataset
Project menyediakan dataset demonstrasi sintetis:
- 480 baris
- 6 produk
- 80 periode bulanan per produk
- Januari 2020 sampai Agustus 2026
Dataset sintetis digunakan untuk memastikan seluruh pipeline dapat didemonstrasikan tanpa bergantung pada data eksternal.

## 7. Fitur yang Digunakan
Fitur numerik: year, month, day, previous_price, demand.
Fitur kategorikal: product, category.
Target: price.
Tanggal diubah menjadi fitur tahun, bulan, dan hari. Fitur kategorikal diproses menggunakan One-Hot Encoding. Missing value numerik ditangani melalui median pada tahap training.

## 8. Preprocessing
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
Penanganan duplikat/konflik
 ↓
Penanganan missing value
 ↓
Feature engineering
 ↓
Data siap training
```
Sistem juga memiliki mekanisme mapping kolom sehingga CSV dengan nama seperti `tanggal`, `nama_barang`, dan `harga` dapat dipetakan ke schema internal.

## 9. Pembagian Data
Data tidak dibagi secara acak. Sistem menggunakan pembagian berdasarkan urutan waktu agar periode lebih baru digunakan sebagai data pengujian.

```text
Periode lama                         Periode baru
|-------------------------------------------|
          TRAIN                 TEST
            80%                  20%
```

## 10. Model Machine Learning
### Random Forest Regressor
Model utama terdiri dari banyak decision tree. Setiap tree menghasilkan prediksi dan hasilnya diagregasi menjadi prediksi akhir.
Konfigurasi utama: `n_estimators=160`, `max_depth=12`, `min_samples_leaf=2`, `random_state=42`.

## 11. Model Pembanding
### Linear Regression
Model pembanding sederhana.
### Baseline
Baseline menggunakan harga sebelumnya sebagai estimasi harga berikutnya.

## 12. Evaluasi
**MAE:** rata-rata besar kesalahan absolut.
**MSE:** rata-rata kuadrat kesalahan.
**RMSE:** akar dari MSE dan berada pada satuan harga.
**R²:** mengukur seberapa besar variasi target dapat dijelaskan model pada data pengujian.

## 13. Hasil Eksperimen Dataset Demo
| Model | MAE | RMSE | R² |
|---|---:|---:|---:|
| Random Forest | Rp522,66 | Rp777,78 | 0,9949 |
| Linear Regression | Rp530,85 | Rp810,86 | 0,9945 |
| Baseline | Rp630,21 | Rp950,63 | 0,9924 |

Interpretasi: pada eksperimen dataset sintetis tersebut, Random Forest menghasilkan error pengujian paling rendah di antara tiga pendekatan yang dibandingkan. Nilai tersebut tidak boleh dianggap sebagai performa untuk seluruh harga pasar karena dataset demo bersifat sintetis.

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
src/train.py            → training, evaluasi, penyimpanan model
src/predict.py          → prediksi data baru
src/generate_data.py    → pembuatan dataset demo sintetis
```

## 16. Alur Penggunaan Aplikasi
**Dashboard:** jumlah data, jumlah produk, harga minimum/maksimum, dan grafik tren.
**Dataset:** melihat dataset aktif dan statistik validasi.
**Training Model:** menjalankan training dan melihat perbandingan model.
**Prediksi Harga:** memasukkan produk, tanggal, harga sebelumnya, dan demand untuk mendapatkan estimasi.
**Model Comparison:** menampilkan metrik MAE, RMSE, dan R².

## 17. Demo Singkat
```text
1. Jalankan aplikasi
2. Buka Training Model
3. Klik Latih model
4. Lihat hasil evaluasi
5. Buka Prediksi Harga
6. Pilih produk
7. Masukkan harga sebelumnya
8. Masukkan demand
9. Klik Prediksi
10. Tampilkan estimasi harga
```

## 18. Contoh Penjelasan Saat Presentasi
> AI Price Predictor adalah sistem prediksi harga berbasis supervised machine learning. Sistem menggunakan data historis sebagai bahan pembelajaran dan memprediksi harga pada periode berikutnya. Model utama yang digunakan adalah Random Forest Regressor.

> Sebelum training, data melalui preprocessing dan feature engineering. Data kemudian dibagi berdasarkan urutan waktu menjadi training dan testing. Random Forest dibandingkan dengan Linear Regression dan baseline harga sebelumnya.

> Setelah model dilatih, pengguna dapat memasukkan data baru melalui aplikasi Streamlit untuk memperoleh estimasi harga. Seluruh proses berjalan secara lokal tanpa API AI eksternal.

## 19. Pertanyaan yang Mungkin Ditanyakan Dosen
**AI-nya di mana?** AI/ML berada pada proses training dan prediksi menggunakan Random Forest Regressor. Streamlit hanya menjadi antarmuka.
**Kenapa Random Forest?** Random Forest dapat menggabungkan banyak decision tree dan menangkap pola hubungan yang tidak harus linear.
**Kenapa bukan classification?** Karena target yang diprediksi adalah nilai harga numerik kontinu.
**Kenapa data dibagi berdasarkan waktu?** Untuk mensimulasikan kondisi belajar dari masa lalu lalu memprediksi periode yang lebih baru.
**Apa baseline?** Pendekatan sederhana menggunakan harga periode sebelumnya.
**Apakah R² 0,9949 berarti akurasi 99,49%?** Tidak. R² bukan accuracy classification.
**Apakah data nyata?** Dataset demo adalah sintetis. Untuk penggunaan nyata diperlukan data historis riil.
**Apakah prediksi menjamin harga pasar?** Tidak. Output merupakan estimasi berdasarkan pola dataset.

## 20. Keterbatasan
1. Dataset demo masih sintetis.
2. Belum memasukkan faktor eksternal seperti inflasi, cuaca, distribusi, promo, atau kebijakan.
3. Kualitas prediksi bergantung pada kualitas dan relevansi data historis.
4. Performa pada dataset demo tidak otomatis mewakili performa pada data dunia nyata.

## 21. Pengembangan Selanjutnya
- Menggunakan dataset harga riil.
- Menambahkan faktor eksternal.
- Menambahkan lebih banyak fitur historis.
- Hyperparameter tuning.
- Membandingkan algoritma tambahan.
- Grafik aktual vs prediksi.
- Histori prediksi.
- Interval ketidakpastian.

## 22. Kesimpulan
AI Price Predictor menunjukkan penerapan supervised machine learning pada masalah prediksi harga.

```text
Data Historis
→ Preprocessing
→ Feature Engineering
→ Training
→ Evaluasi
→ Model
→ Prediksi
```

Model utama adalah Random Forest Regressor dan hasil eksperimen dibandingkan dengan Linear Regression serta baseline harga sebelumnya.
Project dapat dijalankan secara lokal menggunakan Python dan Streamlit tanpa API AI eksternal.