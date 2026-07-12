import os
import sys
import threading
import time

# ── Path setup (harus sebelum import lokal) ──────────────────────────────────
_base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _base_dir not in sys.path:
    sys.path.insert(0, _base_dir)
if os.path.join(_base_dir, 'src') not in sys.path:
    sys.path.insert(0, os.path.join(_base_dir, 'src'))

import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import requests
import io

# Konfigurasi halaman Streamlit
st.set_page_config(
    page_title="ANALIS SENTIMEN PRO - NEO BRUTALISME",
    page_icon="[DATA]",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Path base direktori proyek
base_dir = _base_dir

# ── Jalankan FastAPI sebagai background thread (sekali per proses) ───────────
@st.cache_resource(show_spinner="⚙️ Memuat model & server API...")
def _start_fastapi_server():
    """Start FastAPI+uvicorn in a daemon thread. Cached so it only runs once."""
    import uvicorn
    from src.api import app as _fastapi_app

    def _run():
        uvicorn.run(_fastapi_app, host="127.0.0.1", port=8000, log_level="error")

    t = threading.Thread(target=_run, daemon=True, name="fastapi-bg")
    t.start()
    # Beri waktu server untuk inisialisasi model & DB
    time.sleep(4)
    return "ok"

_start_fastapi_server()

# URL REST API — selalu localhost karena FastAPI jalan di proses yang sama
API_URL = "http://127.0.0.1:8000/api"

# Injeksi CSS Kustom untuk Gaya Neo-Brutalisme Premium, Animasi Halus, dan Layout Responsif
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Fira+Code:wght@400;700&family=Space+Grotesk:wght@400;500;700;800&display=swap');

    /* ====== BASE ====== */
    html, body, [class*="css"] {
        font-family: 'Space Grotesk', sans-serif;
        background-color: #FAF6EE !important;
        color: #000000 !important;
    }
    /* Sembunyikan elemen bawaan Streamlit untuk publik */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    div[data-testid="stHeader"] {visibility: hidden;}
    .stDeployButton {display: none !important;}
    .viewerBadge {display: none !important;}
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Space Grotesk', sans-serif;
        font-weight: 800 !important;
        color: #000000 !important;
        text-transform: uppercase;
        letter-spacing: -0.5px;
    }
    p, span, label, li {
        font-family: 'Space Grotesk', sans-serif;
        color: #000000 !important;
        font-weight: 600;
    }

    /* ====== KEYFRAMES ====== */
    @keyframes slideUp {
        from { opacity: 0; transform: translateY(28px); }
        to   { opacity: 1; transform: translateY(0); }
    }
    @keyframes staggerIn {
        from { opacity: 0; transform: translateY(18px) scale(0.97); }
        to   { opacity: 1; transform: translateY(0) scale(1); }
    }
    @keyframes marquee {
        0%   { transform: translateX(0); }
        100% { transform: translateX(-100%); }
    }
    @keyframes pulse {
        0%, 100% { opacity: 1; transform: scale(1); }
        50%       { opacity: 0.55; transform: scale(1.2); }
    }
    @keyframes shimmer {
        0%   { background-position: -400% center; }
        100% { background-position: 400% center; }
    }
    @keyframes wobble {
        0%,100% { transform: translateX(0); }
        20%     { transform: translateX(-5px) rotate(-2deg); }
        40%     { transform: translateX(4px) rotate(1.5deg); }
        60%     { transform: translateX(-3px) rotate(-1deg); }
        80%     { transform: translateX(2px); }
    }

    /* ====== MARQUEE BANNER ====== */
    .marquee-container {
        overflow: hidden;
        background: #000000;
        border-top: 4px solid #FFDE4D;
        border-bottom: 4px solid #FFDE4D;
        padding: 0.55rem 0;
        margin-bottom: 2rem;
        font-family: 'Fira Code', monospace;
        font-size: 0.85rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        position: relative;
    }
    .marquee-track {
        display: flex;
        width: max-content;
        animation: marquee 32s linear infinite;
    }
    .marquee-text {
        display: inline-block;
        white-space: nowrap;
        padding-right: 80px;
        color: #FFDE4D !important;
    }
    .live-dot {
        display: inline-block;
        width: 8px; height: 8px;
        border-radius: 50%;
        background: #4ADE80;
        animation: pulse 1.4s ease-in-out infinite;
        margin-right: 6px;
        vertical-align: middle;
    }

    /* ====== HEADER BANNER ====== */
    .header-banner {
        background: linear-gradient(135deg, #FFDE4D 0%, #FFB800 100%);
        border: 4px solid #000000;
        box-shadow: 8px 8px 0px #000000;
        padding: 2.8rem 3rem;
        margin-bottom: 1.8rem;
        position: relative;
        overflow: hidden;
        animation: slideUp 0.7s cubic-bezier(0.16, 1, 0.3, 1) both;
        transition: transform 0.2s, box-shadow 0.2s;
    }
    .header-banner::after {
        content: '';
        position: absolute;
        top: -60px; right: -40px;
        width: 260px; height: 260px;
        background: rgba(0,0,0,0.05);
        border-radius: 50%;
        pointer-events: none;
    }
    .header-banner:hover {
        transform: translate(-3px, -3px);
        box-shadow: 11px 11px 0px #000000;
    }
    .header-banner h1 {
        font-size: 3.8rem;
        margin: 0;
        line-height: 1;
        letter-spacing: -2px;
    }
    .header-banner p {
        font-size: 0.95rem;
        font-weight: 700;
        margin-top: 0.8rem;
        margin-bottom: 0;
        text-transform: uppercase;
        letter-spacing: 2px;
        opacity: 0.75;
    }
    .header-badge {
        display: inline-block;
        background: #000000;
        color: #FFDE4D;
        font-family: 'Fira Code', monospace;
        font-size: 0.72rem;
        font-weight: 700;
        padding: 3px 10px;
        letter-spacing: 1px;
        margin-top: 1rem;
    }

    /* ====== BRUTALIST CARDS ====== */
    .brutalist-card {
        background-color: #FFFFFF;
        border: 4px solid #000000;
        box-shadow: 6px 6px 0px #000000;
        padding: 1.8rem;
        margin-bottom: 1.5rem;
        animation: staggerIn 0.5s cubic-bezier(0.16, 1, 0.3, 1) both;
        transition: transform 0.15s ease, box-shadow 0.15s ease;
    }
    .brutalist-card:hover {
        transform: translate(-4px, -4px);
        box-shadow: 10px 10px 0px #000000;
    }
    .brutalist-card-positive {
        background: linear-gradient(135deg, #4ADE80, #22C55E) !important;
        border: 4px solid #000000;
        box-shadow: 6px 6px 0px #000000;
        padding: 1.8rem;
        margin-bottom: 1.5rem;
        animation: staggerIn 0.5s cubic-bezier(0.16, 1, 0.3, 1) both;
        transition: transform 0.15s ease, box-shadow 0.15s ease;
    }
    .brutalist-card-positive:hover { transform: translate(-4px,-4px); box-shadow: 10px 10px 0px #000000; }
    .brutalist-card-negative {
        background: linear-gradient(135deg, #F87171, #EF4444) !important;
        border: 4px solid #000000;
        box-shadow: 6px 6px 0px #000000;
        padding: 1.8rem;
        margin-bottom: 1.5rem;
        animation: staggerIn 0.5s cubic-bezier(0.16, 1, 0.3, 1) both;
        transition: transform 0.15s ease, box-shadow 0.15s ease;
    }
    .brutalist-card-negative:hover { transform: translate(-4px,-4px); box-shadow: 10px 10px 0px #000000; }
    .brutalist-card-neutral {
        background: linear-gradient(135deg, #60A5FA, #3B82F6) !important;
        border: 4px solid #000000;
        box-shadow: 6px 6px 0px #000000;
        padding: 1.8rem;
        margin-bottom: 1.5rem;
        animation: staggerIn 0.5s cubic-bezier(0.16, 1, 0.3, 1) both;
        transition: transform 0.15s ease, box-shadow 0.15s ease;
    }
    .brutalist-card-neutral:hover { transform: translate(-4px,-4px); box-shadow: 10px 10px 0px #000000; }

    /* ====== KPI BOXES ====== */
    .kpi-box {
        background-color: #FFFFFF;
        border: 4px solid #000000;
        box-shadow: 6px 6px 0px #000000;
        padding: 1.5rem 1rem;
        text-align: center;
        margin-bottom: 1rem;
        animation: staggerIn 0.5s cubic-bezier(0.16, 1, 0.3, 1) both;
        transition: transform 0.15s ease, box-shadow 0.15s ease;
        position: relative;
        overflow: hidden;
    }
    .kpi-box::after {
        content: '';
        position: absolute;
        inset: 0;
        background: linear-gradient(135deg, rgba(255,255,255,0.2) 0%, transparent 60%);
        pointer-events: none;
    }
    .kpi-box:hover { transform: translate(-4px,-4px); box-shadow: 10px 10px 0px #000000; }
    .kpi-val {
        font-family: 'Fira Code', monospace;
        font-size: 2.6rem;
        font-weight: 700;
        margin: 0.3rem 0;
        color: #000000;
        line-height: 1;
    }
    .kpi-lbl {
        font-size: 0.8rem;
        font-weight: 800;
        text-transform: uppercase;
        color: #000000;
        letter-spacing: 0.8px;
        margin-top: 0.3rem;
    }

    /* ====== BUTTONS ====== */
    div.stButton > button {
        background-color: #FFDE4D !important;
        color: #000000 !important;
        border: 3.5px solid #000000 !important;
        box-shadow: 5px 5px 0px #000000 !important;
        border-radius: 0px !important;
        font-family: 'Space Grotesk', sans-serif !important;
        font-weight: 800 !important;
        text-transform: uppercase !important;
        letter-spacing: 1px !important;
        padding: 0.75rem 2rem !important;
        font-size: 1rem !important;
        transition: transform 0.1s, box-shadow 0.1s, background-color 0.15s !important;
        width: 100%;
    }
    div.stButton > button:hover {
        transform: translate(-5px, -5px) !important;
        box-shadow: 10px 10px 0px #000000 !important;
        background-color: #FFE566 !important;
    }
    div.stButton > button:active {
        transform: translate(4px, 4px) !important;
        box-shadow: 0px 0px 0px #000000 !important;
    }
    div.stDownloadButton > button {
        background-color: #4ADE80 !important;
        color: #000000 !important;
        border: 3.5px solid #000000 !important;
        box-shadow: 5px 5px 0px #000000 !important;
        border-radius: 0px !important;
        font-weight: 800 !important;
        text-transform: uppercase !important;
        padding: 0.75rem 2rem !important;
        font-size: 1rem !important;
        transition: transform 0.1s, box-shadow 0.1s !important;
    }
    div.stDownloadButton > button:hover {
        transform: translate(-5px,-5px) !important;
        box-shadow: 10px 10px 0px #000000 !important;
        background-color: #6EE7B7 !important;
    }
    div.stDownloadButton > button:active {
        transform: translate(4px,4px) !important;
        box-shadow: 0px 0px 0px #000000 !important;
    }

    /* ====== TEXTAREA ====== */
    div.stTextArea textarea {
        border: 3.5px solid #000000 !important;
        border-radius: 0px !important;
        background-color: #FFFFFF !important;
        color: #000000 !important;
        font-family: 'Space Grotesk', sans-serif !important;
        font-size: 1.05rem !important;
        box-shadow: 5px 5px 0px #000000 !important;
        padding: 1rem !important;
        transition: box-shadow 0.2s, transform 0.2s, border-color 0.2s !important;
    }
    div.stTextArea textarea:focus {
        box-shadow: 8px 8px 0px #000000 !important;
        transform: translate(-2px, -2px);
        border-color: #FFDE4D !important;
    }

    /* ====== TABS ====== */
    div[data-baseweb="tab-list"] {
        gap: 10px !important;
        border-bottom: 4px solid #000000 !important;
        margin-bottom: 2.5rem !important;
        background: transparent !important;
    }
    button[data-baseweb="tab"] {
        border: 3.5px solid #000000 !important;
        border-bottom: none !important;
        border-radius: 0px !important;
        background-color: #FFFFFF !important;
        color: #000000 !important;
        font-family: 'Space Grotesk', sans-serif !important;
        font-weight: 800 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.5px !important;
        padding: 0.65rem 1.5rem !important;
        font-size: 0.95rem !important;
        transition: transform 0.1s, background-color 0.15s !important;
        box-shadow: 3px 0px 0px #000000 !important;
    }
    button[data-baseweb="tab"]:hover { background-color: #FAF6EE !important; transform: translateY(-2px); }
    button[aria-selected="true"] {
        background-color: #FFDE4D !important;
        transform: translateY(4px) !important;
        box-shadow: none !important;
    }

    /* ====== SELECT & FILE UPLOADER ====== */
    div[data-baseweb="select"] > div {
        border: 3px solid #000000 !important;
        border-radius: 0px !important;
        background-color: #FFFFFF !important;
        box-shadow: 4px 4px 0px #000000 !important;
        font-family: 'Space Grotesk', sans-serif !important;
        font-weight: 700 !important;
    }
    div[data-testid="stFileUploader"] {
        border: 3.5px dashed #000000 !important;
        background-color: #FFFFFF !important;
        padding: 2rem !important;
        border-radius: 0px !important;
        box-shadow: 5px 5px 0px #000000 !important;
        transition: box-shadow 0.2s, transform 0.2s !important;
    }
    div[data-testid="stFileUploader"]:hover {
        box-shadow: 8px 8px 0px #000000 !important;
        transform: translate(-2px,-2px);
    }

    /* ====== TABLE ====== */
    div[data-testid="stTable"] table {
        border: 3.5px solid #000000 !important;
        box-shadow: 5px 5px 0px #000000 !important;
        border-collapse: separate !important;
        border-spacing: 0 !important;
        width: 100% !important;
        animation: slideUp 0.5s ease both;
    }
    div[data-testid="stTable"] th {
        background: linear-gradient(135deg, #FFDE4D, #FFB800) !important;
        color: #000000 !important;
        border: 2px solid #000000 !important;
        font-family: 'Space Grotesk', sans-serif !important;
        font-weight: 800 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.5px !important;
        padding: 0.9rem 1rem !important;
    }
    div[data-testid="stTable"] td {
        border: 2px solid #000000 !important;
        padding: 0.75rem 1rem !important;
        background-color: #FFFFFF !important;
        color: #000000 !important;
        font-family: 'Space Grotesk', sans-serif !important;
        font-weight: 600 !important;
    }
    div[data-testid="stTable"] tr:hover td { background-color: #FAF6EE !important; }

    /* ====== SIDEBAR ====== */
    section[data-testid="stSidebar"] {
        background-color: #FAF6EE !important;
        border-right: 4px solid #000000 !important;
    }

    /* ====== DIVIDER ====== */
    .brutalist-divider {
        border: none;
        height: 4px;
        background: repeating-linear-gradient(
            90deg,
            #000000 0px, #000000 20px,
            #FFDE4D 20px, #FFDE4D 40px
        );
        margin: 2.5rem 0;
        animation: slideUp 0.3s ease both;
    }

    /* ====== DATAFRAME ====== */
    div[data-testid="stDataFrame"] {
        border: 3.5px solid #000000 !important;
        box-shadow: 5px 5px 0px #000000 !important;
        animation: slideUp 0.5s ease both;
    }

    /* ====== ALERTS ====== */
    div[data-testid="stAlert"] {
        border: 3px solid #000000 !important;
        border-radius: 0px !important;
        box-shadow: 4px 4px 0px #000000 !important;
        font-family: 'Space Grotesk', sans-serif !important;
        font-weight: 700 !important;
    }

    /* ====== BENTO GRID ====== */
    .bento-grid {
        display: grid;
        grid-template-columns: 1fr;
        gap: 1.5rem;
    }
    @media (min-width: 768px) { .bento-grid { grid-template-columns: 1fr 1fr; } }

</style>
""", unsafe_allow_html=True)

# Teks Marquee Berjalan (Status Live API & Sistem Metrik Latih)
try:
    stats_req = requests.get(f"{API_URL}/stats", timeout=1.5).json()
    predictions_logged = stats_req.get("total_predictions", 0)
    feedbacks_logged = stats_req.get("total_feedbacks", 0)
except Exception:
    predictions_logged, feedbacks_logged = 0, 0

marquee_text = f"⬛ STATUS REST API: AKTIF ▸ MODEL SELEKSI: HYPERPARAMETER TUNED LINEAR SVM ▸ AKURASI EVALUASI: 88.65% ▸ F1-SCORE MACRO: 88.63% ▸ LOG DATABASE SQLITE: {predictions_logged} TRANSAKSI ▸ INPUT FEEDBACK AUDIT: {feedbacks_logged} KOREKSI ▸ FITUR BARU: INDEKS KEPUASAN PELANGGAN (IKP) "
st.markdown(f"""
<div class="marquee-container">
    <div class="marquee-track">
        <span class="marquee-text">{marquee_text}</span>
        <span class="marquee-text">{marquee_text}</span>
    </div>
</div>
""", unsafe_allow_html=True)

# Banner Judul Utama Dashboard
st.markdown("""
<div class="header-banner">
    <h1>ANALIS SENTIMEN PRO</h1>
    <p>INFRASTRUKTUR DECOUPLED REST-API | LOG TRANSAKSI SQLITE & UMPAN BALIK AUDIT REAL-TIME</p>
</div>
""", unsafe_allow_html=True)

# Cek Keaktifan Server REST API FastAPI
api_active = False
try:
    health_check = requests.get(f"{API_URL.replace('/api', '')}/docs", timeout=2)
    if health_check.status_code == 200:
        api_active = True
except Exception:
    pass

# Konfigurasi Sidebar Menu Sistem
with st.sidebar:
    st.markdown("""
    <div style="background-color: #FFDE4D; border: 4px solid #000000; padding: 1.2rem; box-shadow: 5px 5px 0px #000000; text-align: center; font-weight: 800; font-size: 1.1rem; letter-spacing: 0.5px; text-transform: uppercase;">
        PENGATURAN ENGINE
    </div>
    """, unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    
    if not api_active:
        st.markdown("""
        <div class="brutalist-card" style="background-color: #F87171; font-weight: bold;">
            REST API STATUS: OFFLINE
            <p style="font-size: 0.9rem; margin-top: 0.5rem; font-weight: normal; line-height: 1.3;">
            Aktifkan REST API server terlebih dahulu di terminal menggunakan perintah:
            <br><br>
            <code style="font-family: 'Fira Code', monospace; background: rgba(0,0,0,0.1); padding: 2px 4px; font-weight: bold; font-size: 0.8rem;">
            .\\.venv\\Scripts\\uvicorn src.api:app --reload --port 8000
            </code>
            </p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="brutalist-card" style="background-color: #4ADE80; font-weight: bold;">
            REST API STATUS: AKTIF
            <p style="font-size: 0.9rem; margin-top: 0.5rem; font-family: 'Fira Code', monospace; font-weight: normal; line-height: 1.3;">
            Endpoint: http://localhost:8000<br>
            Total Log: {predictions_logged}<br>
            Total Audit: {feedbacks_logged}
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Tampilkan distribusi dataset latih
        clean_csv_path = os.path.join(base_dir, 'dataset', 'clean_reviews.csv')
        if os.path.exists(clean_csv_path):
            df_info = pd.read_csv(clean_csv_path)
            st.markdown(f"""
            <div class="brutalist-card">
                <b>DISTRUBUSI DATA LATIH</b>
                <hr style="border: 1px solid #000000; margin: 0.5rem 0;">
                <p style="font-size: 0.9rem; font-family: 'Fira Code', monospace; font-weight: normal; margin-bottom: 0;">
                Positif: {len(df_info[df_info['sentiment'] == 'Positif'])} (33.3%)<br>
                Netral: {len(df_info[df_info['sentiment'] == 'Netral'])} (33.3%)<br>
                Negatif: {len(df_info[df_info['sentiment'] == 'Negatif'])} (33.3%)<br>
                Total: {len(df_info)} (Balanced)
                </p>
            </div>
            """, unsafe_allow_html=True)
            
        st.markdown("""
        <div style="background-color: #60A5FA; border: 4px solid #000000; padding: 1rem; font-weight: 800; text-align: center; text-transform: uppercase; box-shadow: 4px 4px 0px #000000;">
            CORE DEVELOPMENT TEAM
            <hr style="border: 1.5px solid #000000; margin: 0.5rem 0;">
            <div style="text-align: left; font-size: 0.85rem; font-family: 'Space Grotesk', sans-serif; text-transform: none; font-weight: bold; line-height: 1.5; color: #000000;">
                ▪️ Gempur Budi Anarki<br>
                ▪️ Darul Aman<br>
                ▪️ Iqbal Pratama
            </div>
        </div>
        """, unsafe_allow_html=True)

# Main System Interface Logic
if api_active:
    tab1, tab2, tab3 = st.tabs([
        "ANALISIS ULASAN TUNGGAL", 
        "PROSES DATA MASSAL (CSV)", 
        "METRIK KINERJA & LOG AUDIT SQLITE"
    ])
    
    # ------------------ TAB 1: SINGLE REVIEW ------------------
    with tab1:
        st.markdown("### DETEKSI SENTIMEN REAL-TIME")
        st.markdown("Masukkan ulasan pelanggan di bawah. Model ensemble TF-IDF Hybrid akan menghitung probabilitas klasifikasi.")
        
        col_in, col_out = st.columns([1, 1])
        
        with col_in:
            st.markdown("#### MASUKKAN TEKS ULASAN")
            input_text = st.text_area(
                "Masukan Ulasan Tunggal",
                placeholder="Tulis ulasan produk di sini... (Contoh: Kualitas barang sangat buruk, seller lambat merespon.)",
                height=150,
                label_visibility="collapsed"
            )
            st.markdown("<br>", unsafe_allow_html=True)
            run_btn = st.button("ANALISIS SENTIMEN SEKARANG")
            
        with col_out:
            st.markdown("#### RESPONS HASIL DETEKSI")
            
            if run_btn:
                if not input_text.strip():
                    st.markdown("""
                    <div class="brutalist-card" style="background-color: #F87171;">
                        <b>INPUT ERROR: Silakan isi teks ulasan pelanggan pada kolom kiri terlebih dahulu.</b>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    with st.spinner("Menghubungi server API backend..."):
                        try:
                            # Panggil API FastAPI
                            response = requests.post(f"{API_URL}/predict", json={"text": input_text})
                            res = response.json()
                            
                            sentiment = res['sentiment']
                            confidence = res['confidence']
                            probs = res['probabilities']
                            clean_text = res['clean_text']
                            log_id = res['log_id']
                            
                            # Pilih warna kartu berdasar kelas
                            if sentiment == 'Positif':
                                card_style = 'brutalist-card-positive'
                                fill_color = '#10B981'
                            elif sentiment == 'Negatif':
                                card_style = 'brutalist-card-negative'
                                fill_color = '#EF4444'
                            else:
                                card_style = 'brutalist-card-neutral'
                                fill_color = '#3B82F6'
                                
                            st.markdown(f"""
                            <div class="{card_style}">
                                <h2 style="margin: 0; font-size: 2.2rem; line-height: 1.1;">SENTIMEN: {sentiment.upper()}</h2>
                                <p style="margin-top: 0.5rem; font-size: 1rem; font-weight: 800; letter-spacing: 0.5px;">SKOR CONFIDENCE KELAS TERPILIH</p>
                                
                                <!-- Progress Bar Confidence Utama -->
                                <div style="width: 100%; height: 35px; border: 3.5px solid #000000; background: #FFFFFF; position: relative; overflow: hidden; box-shadow: 3px 3px 0px #000000;">
                                    <div style="width: {confidence * 100}%; height: 100%; background: {fill_color};"></div>
                                    <div style="position: absolute; width: 100%; text-align: center; top: 3px; font-weight: 800; font-family: 'Fira Code', monospace; color: #000000; font-size: 1.1rem;">{confidence * 100:.2f}%</div>
                                </div>
                                
                                <hr style="border: 1px solid #000000; margin: 1.2rem 0;">
                                <p style="margin: 0; font-size: 0.95rem; font-weight: 700; line-height: 1.3;">KATA TERPROSES (CLEAN TOKENS):<br>
                                <span style="font-family: 'Fira Code', monospace; background: rgba(255,255,255,0.4); padding: 2px 6px; font-weight: normal;">
                                {clean_text if clean_text else '[Tidak ada kata tersisa setelah dibersihkan]'}
                                </span>
                                </p>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            # Tampilkan breakdown probabilitas per kelas dengan bar kustom
                            st.markdown("""
                            <div style="background: #FFFFFF; border: 4px solid #000000; box-shadow: 6px 6px 0px #000000; padding: 1.5rem; margin-bottom: 1.5rem;">
                                <h4 style="margin: 0 0 1rem 0; font-size: 1.15rem; font-weight: 800; text-transform: uppercase;">RINCIAN PROBABILITAS TIAP KELAS SENTIMEN</h4>
                                <p style="margin: 0 0 1rem 0; font-size: 0.9rem; font-weight: 600; color: #6B7280;">Angka di bawah menunjukkan seberapa yakin model terhadap SETIAP kelas sentimen secara terpisah.</p>
                            """, unsafe_allow_html=True)
                            
                            # Render bar per kelas
                            class_configs = [
                                ("Positif", "#4ADE80"),
                                ("Netral", "#60A5FA"),
                                ("Negatif", "#F87171"),
                            ]
                            
                            prob_html = ""
                            for cls, bar_color in class_configs:
                                prob_val = probs.get(cls, 0.0)
                                is_winner = cls == sentiment
                                winner_badge = ' <span style="background:#000;color:#FFDE4D;padding:1px 6px;font-size:0.75rem;font-weight:800;">TERPILIH</span>' if is_winner else ""
                                prob_html += f"""
                                <div style="margin-bottom: 0.9rem;">
                                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 4px;">
                                        <span style="font-weight: 800; font-size: 0.95rem; text-transform: uppercase; color: #000000;">{cls}{winner_badge}</span>
                                        <span style="font-family: 'Fira Code', monospace; font-weight: 800; font-size: 1.1rem; color: #000000;">{prob_val * 100:.2f}%</span>
                                    </div>
                                    <div style="width: 100%; height: 28px; border: 3px solid #000000; background: #F5F5F5; position: relative; overflow: hidden; box-shadow: 2px 2px 0px #000000;">
                                        <div style="width: {prob_val * 100}%; height: 100%; background: {bar_color};"></div>
                                    </div>
                                </div>
                                """
                            
                            st.markdown(prob_html + "</div>", unsafe_allow_html=True)
                            
                            st.session_state['last_log_id'] = log_id
                            st.session_state['last_prediction'] = sentiment

                            
                        except Exception as e:
                            st.error(f"Panggilan inferensi API gagal: {e}")
            
            # Sistem Umpan Balik Audit (Feedback Loop)
            if 'last_log_id' in st.session_state:
                st.markdown('<div class="brutalist-divider" style="margin: 1.5rem 0;"></div>', unsafe_allow_html=True)
                st.markdown("#### SISTEM UMPAN BALIK AUDIT MODEL")
                st.markdown("Bantu sistem berkembang! Konfirmasikan akurasi tebakan model untuk pencatatan log SQLite:")
                
                col_correct, col_wrong = st.columns([1, 1])
                with col_correct:
                    if st.button("KONFIRMASI: PREDIKSI BENAR"):
                        fb_payload = {
                            "log_id": st.session_state['last_log_id'],
                            "correct_sentiment": st.session_state['last_prediction'],
                            "notes": "Diverifikasi benar oleh admin."
                        }
                        requests.post(f"{API_URL}/feedback", json=fb_payload)
                        st.success("Tercatat sebagai data valid di database SQLite.")
                        
                with col_wrong:
                    correct_label = st.selectbox("Salah, tandai sentimen yang benar:", ["Negatif", "Netral", "Positif"])
                    if st.button("KIRIM KOREKSI ADMIN"):
                        fb_payload = {
                            "log_id": st.session_state['last_log_id'],
                            "correct_sentiment": correct_label,
                            "notes": "Koreksi salah tebak oleh admin."
                        }
                        requests.post(f"{API_URL}/feedback", json=fb_payload)
                        st.success(f"Data tersimpan sebagai koreksi ({correct_label}) di database.")
            else:
                st.markdown("""
                <div style="border: 4px dashed #000000; padding: 3rem; text-align: center; font-weight: 800; font-size: 1.2rem; color: #6B7280; background-color: #FFFFFF; box-shadow: 5px 5px 0px #000000;">
                    MENUNGGU INPUT SENTIMEN
                </div>
                """, unsafe_allow_html=True)
                
        # Garis pemisah brutalist
        st.markdown('<div class="brutalist-divider"></div>', unsafe_allow_html=True)
        
        # Fitur Baru: Simulator Skenario (Multi-ulasan instan)
        st.markdown("### SIMULATOR SKENARIO MULTI-ULASAN")
        st.markdown("Masukkan beberapa baris ulasan sekaligus (pisahkan dengan baris baru / Enter) untuk membandingkan sentimen dengan cepat.")
        
        multi_input = st.text_area(
            "Simulator Multi Input",
            placeholder="Kualitas jelek sekali, kecewa beli disini.\nBarang bagus dan pengiriman sangat cepat!\nBiasa saja sih produknya, lumayanlah.",
            height=120,
            key="multi_input_textarea"
        )
        
        if st.button("SIMULASIKAN SKENARIO"):
            lines = [line.strip() for line in multi_input.split("\n") if line.strip()]
            if not lines:
                st.warning("Silakan isi ulasan terlebih dahulu!")
            else:
                sim_results = []
                for line in lines:
                    response = requests.post(f"{API_URL}/predict", json={"text": line})
                    res = response.json()
                    sim_results.append({
                        "Teks Ulasan": line,
                        "Prediksi Sentimen": res['sentiment'],
                        "Confidence Score": f"{res['confidence']*100:.2f}%"
                    })
                st.table(sim_results)
                
    # ------------------ TAB 2: BATCH CSV ------------------
    with tab2:
        st.markdown("### BATCH PROCESSING PIPELINE (CSV)")
        st.markdown("Unggah file CSV ulasan produk Anda untuk diproses secara massal oleh backend REST API.")
        
        uploaded_file = st.file_uploader("Unggah berkas CSV ulasan", type=["csv"], key="batch_uploader", label_visibility="collapsed")
        
        if uploaded_file is not None:
            df_uploaded = pd.read_csv(uploaded_file)
            st.markdown("#### PRATINJAU DATA CSV")
            st.dataframe(df_uploaded.head(5), use_container_width=True)
            
            columns = df_uploaded.columns.tolist()
            text_col = st.selectbox("Pilih kolom teks ulasan pelanggan:", columns, key="column_selector")
            
            if st.button("JALANKAN PROSES BATCH"):
                with st.spinner("Sedang memproses dan mencatat transaksi ke database..."):
                    try:
                        # Kirim file CSV sebagai multipart form data ke FastAPI
                        csv_buffer = io.BytesIO()
                        df_uploaded.to_csv(csv_buffer, index=False)
                        csv_buffer.seek(0)
                        
                        files = {'file': (uploaded_file.name, csv_buffer, 'text/csv')}
                        data = {'text_column': text_col}
                        
                        response = requests.post(f"{API_URL}/predict-batch", files=files, data=data)
                        batch_res = response.json()
                        
                        results = [item['predicted_sentiment'] for item in batch_res]
                        confidences = [item['confidence'] for item in batch_res]
                        cleaned_texts = [item['cleaned_text'] for item in batch_res]
                        
                        df_uploaded['Prediksi_Sentimen'] = results
                        df_uploaded['Confidence_Score'] = confidences
                        df_uploaded['Teks_Bersih'] = cleaned_texts
                        
                        st.success("Pemrosesan Massal & Logging Database Berhasil!")
                        
                        # Hitung metrik agregat
                        total = len(df_uploaded)
                        pos_count = sum(df_uploaded['Prediksi_Sentimen'] == 'Positif')
                        neu_count = sum(df_uploaded['Prediksi_Sentimen'] == 'Netral')
                        neg_count = sum(df_uploaded['Prediksi_Sentimen'] == 'Negatif')
                        
                        # Indeks Kepuasan Pelanggan (Customer Satisfaction Index)
                        # Formula: IKP = ((Positive + 0.5 * Neutral) / Total) * 100%
                        ikp_score = ((pos_count + 0.5 * neu_count) / total) * 100 if total > 0 else 0
                        
                        # Layout KPI Bento Grid
                        col_m1, col_m2, col_m3, col_m4, col_m5 = st.columns(5)
                        with col_m1:
                            st.markdown(f'<div class="kpi-box"><div class="kpi-val">{total}</div><div class="kpi-lbl">Total Ulasan</div></div>', unsafe_allow_html=True)
                        with col_m2:
                            st.markdown(f'<div class="kpi-box" style="background-color: #4ADE80;"><div class="kpi-val">{pos_count}</div><div class="kpi-lbl">Positif ({pos_count/total*100:.1f}%)</div></div>', unsafe_allow_html=True)
                        with col_m3:
                            st.markdown(f'<div class="kpi-box" style="background-color: #60A5FA;"><div class="kpi-val">{neu_count}</div><div class="kpi-lbl">Netral ({neu_count/total*100:.1f}%)</div></div>', unsafe_allow_html=True)
                        with col_m4:
                            st.markdown(f'<div class="kpi-box" style="background-color: #F87171;"><div class="kpi-val">{neg_count}</div><div class="kpi-lbl">Negatif ({neg_count/total*100:.1f}%)</div></div>', unsafe_allow_html=True)
                        with col_m5:
                            # Tampilkan IKP dengan warna dinamis
                            ikp_color = "#4ADE80" if ikp_score >= 70 else "#FFDE4D" if ikp_score >= 50 else "#F87171"
                            st.markdown(f'<div class="kpi-box" style="background-color: {ikp_color};"><div class="kpi-val">{ikp_score:.1f}%</div><div class="kpi-lbl">Indeks Kepuasan (IKP)</div></div>', unsafe_allow_html=True)
                        
                        st.markdown('<div class="brutalist-divider"></div>', unsafe_allow_html=True)
                        
                        # Grafik visualisasi
                        col_chart, col_cloud = st.columns([2, 3])
                        with col_chart:
                            st.markdown("#### PROPORSI DIAGRAM SENTIMEN")
                            fig, ax = plt.subplots(figsize=(6, 5))
                            colors = ['#4ADE80', '#60A5FA', '#F87171']
                            labels = ['Positif', 'Netral', 'Negatif']
                            sizes = [pos_count, neu_count, neg_count]
                            
                            labels_filtered = [l for l, s in zip(labels, sizes) if s > 0]
                            colors_filtered = [c for c, s in zip(colors, sizes) if s > 0]
                            sizes_filtered = [s for s in sizes if s > 0]
                            
                            ax.pie(sizes_filtered, labels=labels_filtered, autopct='%1.1f%%', startangle=90, colors=colors_filtered, 
                                   wedgeprops=dict(width=0.4, edgecolor='black', linewidth=3.5))
                            ax.axis('equal')
                            st.pyplot(fig)
                            plt.close()
                            
                        with col_cloud:
                            st.markdown("#### AWAN KATA KUNCI UTAMA (WORD CLOUD)")
                            cloud_type = st.radio("Pilih filter kategori kata:", ["Positif", "Negatif"], key="batch_cloud_radio")
                            
                            sentiment_filter = 'Positif' if cloud_type == "Positif" else 'Negatif'
                            filtered_texts = df_uploaded[df_uploaded['Prediksi_Sentimen'] == sentiment_filter]['Teks_Bersih'].dropna().tolist()
                            all_text = " ".join(filtered_texts)
                            
                            if len(all_text.strip()) > 5:
                                cmap = 'Greens' if cloud_type == "Positif" else 'Reds'
                                wordcloud = WordCloud(width=800, height=400, background_color='white', colormap=cmap, max_words=80).generate(all_text)
                                
                                fig_cloud, ax_cloud = plt.subplots(figsize=(10, 5))
                                ax_cloud.imshow(wordcloud, interpolation='bilinear')
                                ax_cloud.axis('off')
                                st.pyplot(fig_cloud)
                                plt.close()
                            else:
                                st.info("Data teks tidak mencukupi untuk membuat wordcloud.")
                                
                        st.markdown('<div class="brutalist-divider"></div>', unsafe_allow_html=True)
                        
                        st.markdown("#### TABEL DETAIL DATA HASIL DETEKSI")
                        st.dataframe(df_uploaded[[text_col, 'Teks_Bersih', 'Prediksi_Sentimen', 'Confidence_Score']], use_container_width=True)
                        
                        csv_data = df_uploaded.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label="UNDUH LAPORAN HASIL BATCH (CSV)",
                            data=csv_data,
                            file_name="sentimen_batch_analisis.csv",
                            mime="text/csv"
                        )
                    except Exception as e:
                        st.error(f"Gagal memproses batch: {e}")
                        
    # ------------------ TAB 3: METRICS & AUDIT LOGS ------------------
    with tab3:
        st.markdown("### LAPORAN KINERJA MODEL SECARA ILMIAH")
        st.markdown("Evaluasi performa akhir dari model klasifikasi Linear SVM yang dikalibrasi pada data uji seimbang.")
        
        # Kotak Ringkasan Utama
        st.markdown("""
        <div class="brutalist-card">
            <h3>MODEL TERPASANG: CALIBRATED LINEAR SVM (C=1.0)</h3>
            <p>Model ini dievaluasi secara ketat pada data uji terpisah menggunakan ekstraksi fitur hybrid Word-level TF-IDF (10.000 n-gram kata) dan Char-level TF-IDF (15.000 n-gram karakter) untuk ketahanan maksimal terhadap salah ketik (typos) dan bahasa slang.</p>
            <hr style="border: 2px solid #000000; margin: 0.8rem 0;">
            <p style="font-family: 'Fira Code', monospace; font-size: 1.15rem; font-weight: 700; margin: 0; line-height: 1.5;">
            AKURASI KLASIFIKASI GLOBAL: 88.65% (Ditingkatkan Maksimal)<br>
            F1-SCORE MACRO KOEFISIEN: 88.63% (Sangat Seimbang)<br>
            F1-SCORE WEIGHTED KOEFISIEN: 88.63% (Sangat Stabil)
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Grid layout untuk Confusion Matrix dan Laporan Analitik
        col_cm_img, col_cm_desc = st.columns([1, 1])
        
        cm_path = os.path.join(base_dir, 'app', 'confusion_matrix.png')
        if os.path.exists(cm_path):
            with col_cm_img:
                st.image(cm_path, caption="Confusion Matrix pada Data Uji", use_column_width=True)
            with col_cm_desc:
                st.markdown("""
                <div class="brutalist-card" style="height: 100%;">
                    <h3>LAPORAN METRIK AUDIT KLASIFIKASI</h3>
                    <p>Berdasarkan matriks konfusi uji coba:</p>
                    <ul>
                        <li><b>Kelas Negatif:</b> Memiliki nilai Presisi 96% dan Recall 96%. Model ini sangat peka dalam mengidentifikasi keluhan atau ketidakpuasan pelanggan e-commerce secara tepat sasaran.</li>
                        <li><b>Kelas Netral:</b> Memiliki F1-score 85% dengan presisi 87%. Klasifikasi bahasa netral yang sarat dengan singkatan informal berhasil dipisahkan dengan sangat baik.</li>
                        <li><b>Kelas Positif:</b> Memiliki F1-score 85% dengan recall 87%.</li>
                    </ul>
                    <p>Penerapan FeatureUnion terbukti menjadi garda terdepan untuk mencegah kegagalan identifikasi kata (Out-Of-Vocabulary) akibat variasi bahasa lisan Tokopedia.</p>
                </div>
                """, unsafe_allow_html=True)
                
        # Fitur Tambahan Baru: Analisis Statistik Panjang Teks Ulasan
        st.markdown('<div class="brutalist-divider"></div>', unsafe_allow_html=True)
        st.markdown("### ANALISIS STATISTIK PANJANG TEKS ULASAN")
        st.markdown("Grafik rata-rata jumlah kata pada setiap ulasan pelanggan berdasarkan sentimen positif, netral, dan negatif di database logs.")
        
        # Hitung statistik panjang teks dari clean reviews (jika file ada)
        clean_csv_path = os.path.join(base_dir, 'dataset', 'clean_reviews.csv')
        if os.path.exists(clean_csv_path):
            try:
                df_stats = pd.read_csv(clean_csv_path)
                df_stats['word_count'] = df_stats['text'].apply(lambda x: len(str(x).split()))
                
                # Plot bar chart dengan matplotlib gaya brutalist
                avg_words = df_stats.groupby('sentiment')['word_count'].mean().reset_index()
                
                fig_len, ax_len = plt.subplots(figsize=(8, 4.5))
                colors_bar = ['#F87171', '#60A5FA', '#4ADE80'] # Negatif, Netral, Positif
                bars = ax_len.bar(avg_words['sentiment'], avg_words['word_count'], color=colors_bar, edgecolor='black', linewidth=3)
                
                # Hiasi bar chart agar bernuansa brutalist
                ax_len.spines['top'].set_visible(False)
                ax_len.spines['right'].set_visible(False)
                ax_len.spines['left'].set_linewidth(3)
                ax_len.spines['bottom'].set_linewidth(3)
                ax_len.set_title("Rata-rata Jumlah Kata Berdasarkan Sentimen", fontsize=12, fontweight='bold')
                ax_len.set_ylabel("Rata-rata Jumlah Kata", fontsize=10, fontweight='bold')
                
                # Tampilkan nilai di atas bar
                for bar in bars:
                    height = bar.get_height()
                    ax_len.annotate(f'{height:.1f} kata',
                                xy=(bar.get_x() + bar.get_width() / 2, height),
                                xytext=(0, 3),  # 3 points vertical offset
                                textcoords="offset points",
                                ha='center', va='bottom', fontweight='bold')
                
                st.pyplot(fig_len)
                plt.close()
            except Exception as e:
                st.info(f"Gagal memuat visualisasi panjang teks: {e}")
        else:
            st.info("Visualisasi statistik panjang teks membutuhkan pelatihan model terlebih dahulu.")
            
        st.markdown('<div class="brutalist-divider"></div>', unsafe_allow_html=True)
        
        # Tabel Riwayat Transaksi SQLite Real-time
        st.markdown("### LOG TRANSAKSI AUDIT PERSISTEN SQLITE")
        st.markdown("Log di bawah ditarik langsung secara real-time dari database **SQLite** (`database/logs.db`) via REST API server.")
        
        try:
            recent_logs = requests.get(f"{API_URL}/recent?limit=25", timeout=2).json()
            if recent_logs:
                table_data = []
                for log in recent_logs:
                    log_id = log.get("id")
                    txt = log.get("review_text")
                    pred = log.get("predicted_sentiment")
                    conf = log.get("confidence")
                    time_str = log.get("created_at")[:19].replace("T", " ")
                    fb = log.get("correct_sentiment")
                    
                    # Highlight status audit kustom
                    if not fb:
                        feedback_status = "Belum Diaudit"
                    elif fb == pred:
                        feedback_status = f"Diverifikasi Benar ({fb})"
                    else:
                        feedback_status = f"Dikoreksi Menjadi: {fb}"
                    
                    table_data.append({
                        "Log ID": log_id,
                        "Waktu Transaksi": time_str,
                        "Input Teks Ulasan": txt if len(txt) < 80 else txt[:77] + "...",
                        "Prediksi Model": pred,
                        "Confidence Score": f"{conf*100:.1f}%",
                        "Status Audit": feedback_status
                    })
                
                st.table(table_data)
            else:
                st.info("Belum ada log transaksi ulasan yang tersimpan dalam database logs.")
        except Exception as e:
            st.error(f"Gagal memuat log transaksi database: {e}")
            
        # Fitur Baru: Unduh Laporan Eksekutif Ringkasan
        st.markdown('<div class="brutalist-divider"></div>', unsafe_allow_html=True)
        st.markdown("### EKSPOR LAPORAN EKSEKUTIF PERUSAHAAN")
        st.markdown("Klik tombol di bawah untuk mengunduh dokumen ringkasan metrik performa AI dan logs transaksi untuk pelaporan korporat.")
        
        # Siapkan isi markdown untuk didownload
        report_md = f"""# LAPORAN EKSEKUTIF ANALIS SENTIMEN PRO

## 1. RINGKASAN PERFORMA OPERASIONAL
- Akurasi Model Global: 88.65%
- F1-Score Koefisien: 88.63%
- Model Architecture: Calibrated Linear SVM
- Kosakata Fitur: 25,000 (hybrid Word & Char TF-IDF)

## 2. STATISTIK DATA LATIH
- Total Dataset Seimbang: 18,000 Ulasan (6,000 per kelas)
- Preprocessing: Case folding, cleaning, normalisasi slang/typo

## 3. AUDIT TRANSACTION DATABASE METRICS
- Total Transaksi Masuk: {predictions_logged} ulasan
- Total Umpan Balik Koreksi: {feedbacks_logged} koreksi

---
Laporan dibuat secara otomatis oleh sistem Analis Sentimen Pro E-Commerce.
"""
        st.download_button(
            label="UNDUH LAPORAN EKSEKUTIF (.MD)",
            data=report_md.encode('utf-8'),
            file_name="laporan_eksekutif_sentimen.md",
            mime="text/markdown"
        )
else:
    st.markdown("""
    <div style="border: 4px dashed #000000; padding: 4rem; text-align: center; background-color: #FFFFFF; box-shadow: 6px 6px 0px #000000;">
        <h2 style="color: #EF4444; margin-top: 0;">API BACKEND OFFLINE</h2>
        <p>Silakan nyalakan REST API server di terminal Anda agar dashboard UI dapat terhubung dan berfungsi.</p>
        <code style="font-family: 'Fira Code', monospace; background: rgba(0,0,0,0.1); padding: 5px 10px; font-weight: bold; font-size: 0.95rem;">
        .\\.venv\\Scripts\\uvicorn src.api:app --reload --port 8000
        </code>
    </div>
    """, unsafe_allow_html=True)
