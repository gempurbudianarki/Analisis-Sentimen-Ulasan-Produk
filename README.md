<div align="center">

# ANALIS SENTIMEN PRO
### Sistem Klasifikasi Sentimen Ulasan Produk E-Commerce Berbasis Machine Learning

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.3+-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![SQLite](https://img.shields.io/badge/SQLite-3-003B57?style=for-the-badge&logo=sqlite&logoColor=white)](https://sqlite.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](LICENSE)

<br>

> **Sistem analisis sentimen tingkat industri** untuk mengklasifikasikan ulasan produk e-commerce Indonesia ke dalam tiga kelas: **Positif, Netral, dan Negatif** menggunakan arsitektur decoupled REST API + dashboard interaktif Neo-Brutalist.

</div>

---

## Daftar Isi

- [Gambaran Proyek](#gambaran-proyek)
- [Arsitektur Sistem](#arsitektur-sistem)
- [Akurasi dan Performa Model](#akurasi-dan-performa-model)
- [Teknologi yang Digunakan](#teknologi-yang-digunakan)
- [Struktur Folder](#struktur-folder)
- [Instalasi dan Pengaturan](#instalasi-dan-pengaturan)
- [Cara Menjalankan](#cara-menjalankan)
- [Dokumentasi API](#dokumentasi-api)
- [Fitur Dashboard](#fitur-dashboard)
- [Alur Sistem Pipeline](#alur-sistem-pipeline)
- [Dataset](#dataset)
- [Kontribusi](#kontribusi)
- [Lisensi](#lisensi)

---

## Gambaran Proyek

**Analis Sentimen Pro** adalah sistem klasifikasi sentimen otomatis yang dirancang untuk membantu bisnis e-commerce membaca dan memahami ribuan ulasan pelanggan secara instan tanpa analisis manual.

### Mengapa Proyek Ini Dibutuhkan

- Perusahaan e-commerce rata-rata menerima ribuan ulasan per hari, mustahil dibaca satu per satu
- Ulasan pelanggan adalah sumber data bisnis paling jujur untuk perbaikan produk
- Sistem manual rentan bias dan tidak konsisten
- Model ML modern mampu memahami bahasa gaul dan typos ulasan Indonesia

### Fitur Utama

| Fitur | Keterangan |
|---|---|
| **Klasifikasi Real-Time** | Prediksi sentimen tunggal dengan confidence score lengkap |
| **Rincian Probabilitas Per Kelas** | Bar visual menunjukkan % probabilitas Positif/Netral/Negatif |
| **Batch Processing CSV** | Analisis ratusan ulasan sekaligus melalui upload file CSV |
| **Simulator Skenario Multi-Ulasan** | Uji beberapa ulasan sekaligus tanpa upload file |
| **Indeks Kepuasan Pelanggan (IKP)** | Metrik korporat agregat berbobot untuk monitoring sentimen |
| **Word Cloud Interaktif** | Visualisasi kata kunci dominan per kategori sentimen |
| **Log Audit SQLite Real-Time** | Setiap prediksi tersimpan permanen di database relasional |
| **Sistem Umpan Balik** | Admin dapat mengonfirmasi atau mengoreksi prediksi model |
| **Ekspor Laporan Eksekutif** | Unduh ringkasan performa model dalam format Markdown |

---

## Arsitektur Sistem

Sistem ini dibangun menggunakan pola arsitektur **decoupled (terpisah)** dimana backend dan frontend berjalan sebagai layanan mandiri yang berkomunikasi via HTTP REST API.

```
+----------------------------------------------------------+
|                    PENGGUNA / ADMIN                       |
+------------------------+---------------------------------+
                         | Browser
                         v
+----------------------------------------------------------+
|              FRONTEND: Streamlit Dashboard                |
|                  (http://localhost:8501)                   |
|  - Input ulasan tunggal    - Visualisasi confidence       |
|  - Upload CSV batch        - Log audit tabel SQLite       |
|  - Simulator multi-ulasan  - Ekspor laporan eksekutif     |
+------------------------+---------------------------------+
                         | HTTP REST API (JSON)
                         v
+----------------------------------------------------------+
|              BACKEND: FastAPI Server                      |
|                  (http://localhost:8000)                   |
|  POST /api/predict          Inferensi tunggal             |
|  POST /api/predict-batch    Batch CSV inference           |
|  POST /api/feedback         Simpan koreksi admin          |
|  GET  /api/recent           Tarik log terbaru             |
|  GET  /api/stats            Statistik sistem              |
+----------+----------------------------+-------------------+
           |                            |
           v                            v
+-------------------+     +-----------------------------+
|   MODEL (.pkl)    |     |    DATABASE: SQLite         |
|  Calibrated       |     |    (database/logs.db)       |
|  Linear SVM C=1.0 |     |  - prediction_logs          |
|  + TF-IDF Hybrid  |     |  - feedback_logs            |
+-------------------+     +-----------------------------+
```

---

## Akurasi dan Performa Model

Model dilatih menggunakan data **18.000 ulasan** e-commerce Indonesia yang seimbang (6.000 per kelas) dengan pendekatan **Grid Search Hyperparameter Tuning** pada Calibrated Linear SVM.

### Hasil Evaluasi pada Data Uji (20%)

| Metrik | Nilai |
|---|---|
| **Akurasi Global** | **88.65%** |
| **F1-Score Macro** | **88.63%** |
| **F1-Score Weighted** | **88.63%** |
| **Presisi Kelas Negatif** | **96%** |
| **F1-Score Kelas Negatif** | **96%** |

### Perbandingan Nilai Parameter C (Grid Search)

| Parameter C | Akurasi | F1-Score Macro |
|---|---|---|
| C = 0.05 | 80.85% | 80.67% |
| C = 0.1 | 82.82% | 82.76% |
| C = 0.2 | 85.33% | 85.29% |
| C = 0.5 | 87.44% | 87.41% |
| **C = 1.0 (Terpilih)** | **88.65%** | **88.63%** |

### Alasan Pemilihan Calibrated Linear SVM

- **Kecepatan inferensi** sangat tinggi (< 5ms per prediksi)
- **Soft probabilitas** via CalibratedClassifierCV menghasilkan confidence score yang kalibratif
- **Feature Union Hybrid TF-IDF** (word + char level) mengatasi typos dan variasi slang secara bersamaan
- Lebih unggul dari ensemble Voting Classifier yang menghasilkan hanya 83.08% akurasi

---

## Teknologi yang Digunakan

### Backend dan Machine Learning

| Library | Versi | Fungsi |
|---|---|---|
| **Python** | 3.10+ | Bahasa pemrograman utama |
| **FastAPI** | 0.100+ | REST API server berperforma tinggi |
| **Uvicorn** | 0.23+ | ASGI server untuk FastAPI |
| **Scikit-learn** | 1.3+ | Model ML, TF-IDF, Pipeline, Grid Search |
| **Sastrawi** | 1.0+ | Stemming bahasa Indonesia |
| **NLTK** | 3.8+ | Tokenisasi dan stopword removal |
| **Pandas** | 2.0+ | Manipulasi data CSV dan analisis |
| **NumPy** | 1.24+ | Operasi array numerik |
| **Joblib** | 1.3+ | Serialisasi dan loading model (.pkl) |
| **SQLite** | Built-in | Database transaksi prediksi persisten |

### Frontend dan Visualisasi

| Library | Versi | Fungsi |
|---|---|---|
| **Streamlit** | 1.30+ | Dashboard web interaktif |
| **Matplotlib** | 3.7+ | Grafik batang dan pie chart |
| **Seaborn** | 0.12+ | Confusion matrix heatmap |
| **WordCloud** | 1.9+ | Visualisasi kata kunci dominan |
| **Requests** | 2.31+ | HTTP client untuk konsumsi REST API |

---

## Struktur Folder

```
Analisis-Sentimen-Ulasan-Produk/
|
+-- app/
|   +-- streamlit_app.py          # Dashboard utama (Frontend Streamlit)
|   +-- confusion_matrix.png      # Visualisasi confusion matrix hasil evaluasi
|
+-- src/
|   +-- preprocessing.py          # Pipeline pembersihan dan normalisasi teks
|   +-- train_model.py            # Pelatihan model + Grid Search hyperparameter
|   +-- api.py                    # Backend REST API server (FastAPI)
|   +-- database.py               # Logika SQLite: logging prediksi + audit feedback
|
+-- dataset/
|   +-- raw_reviews.csv           # Data mentah ulasan Tokopedia
|   +-- clean_reviews.csv         # Data bersih hasil preprocessing pipeline
|
+-- models/
|   +-- model_sentimen.pkl        # Model terlatih: Calibrated SVM + TF-IDF
|
+-- database/
|   +-- logs.db                   # File SQLite: log transaksi prediksi
|
+-- reports/
|   +-- laporan_eksekutif.md      # Laporan ringkasan kinerja sistem
|
+-- BLUEPRINT_Analisis_Sentimen_Ulasan_Produk.md
+-- requirements.txt
+-- README.md
```

---

## Instalasi dan Pengaturan

### Prasyarat

- Python 3.10 atau lebih baru
- Git terinstal di sistem

### Langkah Instalasi

**1. Clone repository:**

```bash
git clone https://github.com/gempurbudianarki/Analisis-Sentimen-Ulasan-Produk.git
cd Analisis-Sentimen-Ulasan-Produk
```

**2. Buat dan aktifkan virtual environment:**

```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# Linux atau macOS
python -m venv .venv
source .venv/bin/activate
```

**3. Install semua dependensi:**

```bash
pip install -r requirements.txt
```

**4. Download resource NLTK:**

```bash
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
```

**5. Latih model (jika belum ada file model_sentimen.pkl):**

```bash
python src/train_model.py
```

---

## Cara Menjalankan

Sistem membutuhkan **dua terminal** yang berjalan bersamaan:

**Terminal 1 - Backend REST API:**

```bash
.venv\Scripts\activate
uvicorn src.api:app --reload --host localhost --port 8000
```

**Terminal 2 - Frontend Dashboard:**

```bash
.venv\Scripts\activate
streamlit run app/streamlit_app.py
```

**Akses sistem setelah kedua server aktif:**

| Komponen | URL |
|---|---|
| Dashboard UI | http://localhost:8501 |
| API Swagger Docs | http://localhost:8000/docs |
| API ReDoc | http://localhost:8000/redoc |

---

## Dokumentasi API

### Base URL: `http://localhost:8000/api`

#### POST /predict - Prediksi Sentimen Tunggal

Request Body:

```json
{
  "text": "Barangnya bagus, pengiriman cepat dan penjual responsif!"
}
```

Response:

```json
{
  "sentiment": "Positif",
  "confidence": 0.8627,
  "probabilities": {
    "Positif": 0.8627,
    "Netral": 0.1140,
    "Negatif": 0.0233
  },
  "clean_text": "barang bagus kirim cepat jual responsif",
  "log_id": 42
}
```

#### POST /predict-batch - Prediksi Batch dari CSV

Request: Multipart form data dengan field:
- `file`: File CSV yang akan diproses
- `text_column`: Nama kolom teks di dalam file CSV

Response: Array JSON berisi hasil prediksi per baris ulasan.

#### POST /feedback - Kirim Koreksi Audit

```json
{
  "log_id": 42,
  "correct_sentiment": "Netral",
  "notes": "Ulasan ini sebenarnya netral, bukan positif."
}
```

#### GET /recent?limit=25 - Log Transaksi Terbaru

Mengembalikan 25 log prediksi terbaru dari database SQLite beserta status audit dan koreksi.

#### GET /stats - Statistik Sistem

```json
{
  "total_predictions": 150,
  "total_feedbacks": 12
}
```

---

## Fitur Dashboard

### Tab 1: Analisis Ulasan Tunggal

- Input textarea untuk memasukkan teks ulasan bebas
- Tombol analisis untuk trigger inferensi real-time ke API
- Kartu hasil dengan warna dinamis (hijau/merah/biru) sesuai kelas sentimen
- Progress bar confidence score kelas terpilih
- Rincian probabilitas tiga kelas dengan bar visual terpisah
- Clean tokens: kata-kata setelah melalui preprocessing pipeline
- Sistem umpan balik audit dengan tombol konfirmasi benar atau koreksi salah

### Tab 2: Proses Data Massal CSV

- Drag and drop atau browse file CSV ulasan
- Selectbox untuk memilih kolom teks yang akan dianalisis
- KPI bento grid: Total / Positif / Netral / Negatif / IKP
- Pie chart proporsi distribusi sentimen
- Word cloud interaktif kata kunci positif atau negatif
- Tabel detail hasil prediksi per baris
- Tombol unduh hasil batch dalam format CSV

### Tab 3: Metrik Kinerja dan Log Audit SQLite

- Ringkasan arsitektur dan performa model
- Confusion matrix heatmap dari data evaluasi
- Laporan metrik per kelas sentimen
- Grafik batang rata-rata panjang kata per kategori sentimen
- Log transaksi real-time ditarik langsung dari SQLite
- Tombol unduh laporan eksekutif format Markdown

---

## Alur Sistem Pipeline

```
Teks Ulasan Mentah
        |
        v
[1] CASE FOLDING          -> Semua huruf menjadi lowercase
        |
        v
[2] TEXT CLEANING         -> Hapus angka, tanda baca, emoji, URL, HTML tag
        |
        v
[3] NORMALISASI SLANG     -> gak->tidak, mntap->mantap, bgs->bagus
        |
        v
[4] TOKENISASI            -> Pecah kalimat menjadi token kata
        |
        v
[5] STOPWORD REMOVAL      -> Hapus kata umum (dan, yang, di, ke, ...)
        |
        v
[6] STEMMING (Sastrawi)   -> membeli->beli, kualitas->kualitas
        |
        v
[7] FEATURE EXTRACTION    -> Word TF-IDF (10k) + Char TF-IDF (15k) = 25k fitur
        |
        v
[8] KLASIFIKASI SVM       -> P(Positif)=86%, P(Netral)=11%, P(Negatif)=3%
        |
        v
[9] OUTPUT + LOGGING      -> Label terpilih + confidence -> Simpan ke SQLite
```

---

## Dataset

| Properti | Detail |
|---|---|
| **Sumber** | Ulasan produk Tokopedia (dataset publik e-commerce Indonesia) |
| **Total sampel** | 18.000 ulasan (setelah class balancing) |
| **Distribusi kelas** | 6.000 Positif / 6.000 Netral / 6.000 Negatif |
| **Format** | CSV dengan kolom `text` dan `sentiment` |
| **Bahasa** | Indonesia termasuk slang dan typos khas marketplace |
| **Pembagian** | 80% training / 20% testing (stratified split) |

---

## Kontribusi

Kontribusi sangat disambut! Berikut langkah-langkahnya:

```bash
# 1. Fork repository ini di GitHub
# 2. Buat branch baru untuk fitur Anda
git checkout -b feature/nama-fitur-anda

# 3. Commit perubahan Anda
git commit -m "feat: tambahkan fitur nama-fitur"

# 4. Push ke branch Anda
git push origin feature/nama-fitur-anda

# 5. Buat Pull Request ke branch main
```

---

## Lisensi

Proyek ini dilisensikan di bawah **MIT License**.

---

<div align="center">

Dibuat untuk tugas akhir mata kuliah **Natural Language Processing (NLP)**

**Gempur Budi Anarki** | Sistem Informasi

</div>
