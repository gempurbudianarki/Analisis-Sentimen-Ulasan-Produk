# Blueprint Project NLP — Analisis Sentimen Ulasan Produk

## 1. Ringkasan Project

**Nama Project:** Analisis Sentimen Ulasan Produk (Product Review Sentiment Analysis)

**Tujuan:** Membangun sistem yang bisa otomatis mengklasifikasikan ulasan produk (review) ke dalam kategori **Positif**, **Negatif**, atau **Netral**, lalu ditampilkan lewat aplikasi sederhana (web/dashboard).

**Kenapa topik ini bagus:**
- Dataset melimpah dan gampang didapat (marketplace, Kaggle, scraping).
- Alur kerja NLP-nya "template klasik" yang jelas tahapannya — cocok buat dinilai dosen.
- Bisa dikembangkan dari simple (ML klasik) sampai advanced (Deep Learning/IndoBERT) tergantung waktu tersisa.
- Ada nilai praktis: bisa dipakai UMKM buat pantau review produk.

---

## 2. Bahasa & Tools yang Dipakai

| Kategori | Pilihan | Keterangan |
|---|---|---|
| Bahasa pemrograman | **Python 3.10+** | Wajib sesuai ketentuan tugas |
| Data handling | **Pandas, NumPy** | Load & bersihin dataset |
| NLP preprocessing | **Sastrawi** (stemming Bahasa Indonesia), **NLTK** (tokenizing, stopwords), **re** (regex) | Sastrawi khusus buat teks Indonesia, wajib kalau reviewnya bahasa Indonesia |
| Feature extraction | **Scikit-learn** (`TfidfVectorizer` / `CountVectorizer`) | Ubah teks jadi angka |
| Model ML klasik | **Scikit-learn** — Naive Bayes / Logistic Regression / SVM | Baseline model |
| Model Deep Learning (opsional, kalau mau nilai lebih) | **TensorFlow/Keras** — LSTM, atau **Transformers (HuggingFace)** — IndoBERT | Model lanjutan |
| Visualisasi | **Matplotlib, Seaborn, WordCloud** | Buat laporan & confusion matrix |
| Aplikasi/interface | **Streamlit** (paling gampang & cepat) atau **Flask** (kalau mau lebih custom) | Sesuai rekomendasi di brief tugas |
| Version control | **Git + GitHub** | Buat kumpulin source code |

> Rekomendasi buat kamu (karena background fullstack): pakai **Streamlit** aja buat cepat, tapi kalau mau pamer skill web bisa juga bikin API pakai **Flask/FastAPI** terus frontend-nya pakai Tailwind kayak project kamu yang lain.

---

## 3. Sumber Dataset

Pilih salah satu (atau gabungan):

1. **Kaggle** — cari "Indonesian product review sentiment" atau "e-commerce review dataset Indonesia"
2. **Scraping sendiri** — review dari Tokopedia/Shopee pakai `BeautifulSoup`/`Selenium` (hati-hati rate limit & ToS)
3. **Dataset publik akademik** — misal dataset ulasan Tokopedia dari repositori riset NLP Bahasa Indonesia (banyak di GitHub, cari "indonesian-sentiment-dataset")

**Format ideal:** CSV dengan minimal 2 kolom → `teks_ulasan`, `label` (positif/negatif/netral). Target minimal ±1000-2000 baris data biar model nggak underfit.

---

## 4. Alur Kerja Project (Sesuai 6 Tahapan Tugas)

### Tahap 1 — Pilih Topik ✅ (Sudah: Analisis Sentimen Ulasan Produk)

### Tahap 2 — Analisis Masalah
- Masalah: Perusahaan/penjual kesulitan membaca ratusan-ribuan review manual satu-satu.
- Solusi: Sistem otomatis klasifikasi sentimen + dashboard ringkasan (persentase positif/negatif, kata yang sering muncul, dll).
- Manfaat: Hemat waktu, bisa deteksi masalah produk dari review negatif secara cepat.

### Tahap 3 — Pengumpulan Data
- Ambil dataset (lihat bagian 3).
- Cek label balance (jangan sampai 90% positif, 10% negatif — bakal bias).
- Simpan sebagai `dataset/raw_reviews.csv`.

### Tahap 4 — Preprocessing Teks
Urutan standar (implementasikan sebagai pipeline/function):
1. **Case folding** — semua huruf jadi lowercase
2. **Cleaning** — hapus angka, tanda baca, emoji, URL, HTML tag
3. **Tokenization** — pecah kalimat jadi kata (`nltk.word_tokenize`)
4. **Stopword removal** — hapus kata umum ("yang", "dan", "di") pakai Sastrawi/NLTK stopwords Indonesia
5. **Stemming** — ubah kata ke bentuk dasar (Sastrawi `StemmerFactory`), misal "membeli" → "beli"
6. (Opsional) **Normalisasi slang** — "ga", "gak", "tdk" → "tidak" (bikin kamus kecil sendiri, penting banget buat review bahasa gaul Indonesia)

### Tahap 5 — Pembangunan Model
**Opsi A (Baseline, wajib ada minimal ini):**
- Feature extraction: `TfidfVectorizer`
- Model: **Multinomial Naive Bayes** atau **Logistic Regression** atau **SVM**
- Split data: 80% train, 20% test

**Opsi B (Advanced, buat nilai plus):**
- Model: **LSTM** (pakai Embedding layer + Keras) atau fine-tuning **IndoBERT**
- Butuh lebih banyak data & waktu training, tapi hasil biasanya lebih akurat

> Saran: kerjain Opsi A dulu sampai jalan penuh (end-to-end), baru kalau waktu masih ada, upgrade ke Opsi B buat dibandingkan performanya di laporan (ini malah nambah poin di rubrik "Ketepatan Penerapan NLP").

### Tahap 6 — Evaluasi & Implementasi
- Hitung **Accuracy, Precision, Recall, F1-score**
- Buat **Confusion Matrix** (visualisasi pakai Seaborn heatmap)
- Bandingkan kalau ada 2 model (ML klasik vs DL)
- Bangun aplikasi Streamlit: input teks review → output prediksi sentimen + confidence score, plus dashboard summary kalau upload banyak review sekaligus (CSV)

---

## 5. Alur Sistem (Diagram Konsep)

```
Input Teks Review
      ↓
Preprocessing (cleaning → tokenizing → stopword removal → stemming)
      ↓
Representasi Teks (TF-IDF / Word Embedding)
      ↓
Model NLP (Naive Bayes/SVM atau LSTM/IndoBERT)
      ↓
Prediksi/Klasifikasi (Positif / Negatif / Netral)
      ↓
Aplikasi (Streamlit Dashboard)
```

---

## 6. Struktur Folder Project (Saran)

```
analisis-sentimen-produk/
├── dataset/
│   ├── raw_reviews.csv
│   └── clean_reviews.csv
├── notebooks/
│   └── eksplorasi_dan_training.ipynb
├── src/
│   ├── preprocessing.py
│   ├── train_model.py
│   └── predict.py
├── models/
│   └── model_sentimen.pkl
├── app/
│   └── streamlit_app.py
├── laporan/
│   └── Laporan_Project_NLP.docx
├── requirements.txt
└── README.md
```

---

## 7. Pembagian Tugas Kelompok (Contoh, sesuaikan jumlah anggota)

| Peran | Tanggung Jawab |
|---|---|
| Data Engineer | Cari/scraping dataset, preprocessing |
| ML Engineer | Bangun & training model, tuning |
| Evaluator | Evaluasi model, bikin confusion matrix & metrics |
| Developer Aplikasi | Bangun aplikasi Streamlit/Flask |
| Dokumentasi | Susun laporan, siapin video demo & slide presentasi |

---

## 8. Checklist Luaran

- [ ] Source code (push ke GitHub, buat repo baru)
- [ ] Dataset yang dipakai (raw + clean)
- [ ] Laporan project (10–15 halaman, format bisa aku bantu bikinin nanti)
- [ ] Video demo (5–10 menit)
- [ ] Slide presentasi (15 menit)

---

## 9. Langkah Selanjutnya

Kalau kamu udah siap eksekusi, aku bisa bantu bikinin:
1. Script preprocessing lengkap (`preprocessing.py`)
2. Script training model baseline (`train_model.py`)
3. Aplikasi Streamlit siap pakai

Tinggal bilang mau mulai dari mana bro.
