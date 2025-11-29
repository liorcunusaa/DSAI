import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import os
import io

# Import modul lokal dari folder src
from src.data_loader import load_local_data
from src.preprocessing import clean_and_handle_missing_values
from src.feature_engineering import create_features
from src.modelling import build_hybrid_model, calculate_evaluation_metrics
from src.integratedRecommender import IntegratedRecommender
from src.evaluasiLlm import LLMTools, HybridEvaluation
from src.visualisasi import run_eda # Kita akan modifikasi run_eda untuk Streamlit

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Definisi Warna Baru
COLOR_PRIMARY = "#385F8C"
COLOR_SECONDARY = "#E9E9E9"

# --- Konfigurasi Global ---
DATA_FILE_PATH = 'data/product_data.csv'

import streamlit as st

st.set_page_config(layout="wide")
# --- Konfigurasi Awal Session State ---
# Inisialisasi status menu kustom (False = Tersembunyi)
if 'menu_visible' not in st.session_state:
    st.session_state.menu_visible = False

# Fungsi untuk membalikkan (toggle) status tampilan menu
def toggle_menu():
    st.session_state.menu_visible = not st.session_state.menu_visible

# Fungsi untuk membuat Header dengan Searchbar dan Tombol Menu
def render_header():
    # Gunakan layout kolom untuk menata elemen dalam header
    # Rasio lebar: 2 (Judul), 8 (Search), 1 (Tombol Tiga Titik)
    col1, col2, col3 = st.columns([2, 8, 1])

    with col1:
        st.markdown("### 🏠 AppName")

    with col2:
        # Searchbar
        st.text_input("Cari Sesuatu...", label_visibility="collapsed", placeholder="Masukkan kata kunci...")

    with col3:
        # **Tombol Simulasi Tiga Titik (Menggunakan st.button)**
        # Ketika tombol diklik, ia memanggil fungsi toggle_menu
        st.button("⋮", key="menu_button", on_click=toggle_menu)

    st.markdown("---") # Garis pemisah

# Fungsi untuk membuat Menu Kustom yang Muncul/Hilang
def render_custom_menu():
    # Cek apakah status 'menu_visible' adalah True
    if st.session_state.menu_visible:
        # Buat container untuk menu yang muncul
        with st.container(border=True):
            st.markdown("#### 🔽 Menu Kustom 🔽")
            # Tata navigasi di dalam menu
            col_nav1, col_nav2, col_nav3 = st.columns(3)
            with col_nav1:
                st.button("📊 Analisis", use_container_width=True)
            with col_nav2:
                st.button("👤 Profil", use_container_width=True)
            with col_nav3:
                st.button("🚪 Keluar", use_container_width=True)
        st.markdown("---")


# Fungsi untuk membuat Footer (Menggunakan CSS untuk posisi tetap)
def render_footer():
    st.markdown(
        """
        <style>
        .footer {
            position: fixed;
            left: 0;
            bottom: 0;
            width: 100%;
            background-color: #e6e6e6; /* Warna latar belakang footer */
            color: black;
            text-align: center;
            padding: 10px;
            font-size: 12px;
            z-index: 100; /* Pastikan footer di atas konten lain */
        }
        </style>
        <div class="footer">
            © 2025 Aplikasi Contoh Streamlit.
        </div>
        """,
        unsafe_allow_html=True
    )

# --- EKSEKUSI TAMPILAN UTAMA ---

# 1. Header (Selalu ditampilkan)
render_header()

# 2. Menu Kustom (Muncul/Hilang berdasarkan klik tombol)
render_custom_menu()

# 3. Konten Utama
st.header("Konten Utama Aplikasi")
st.info("Klik tombol **⋮** di header untuk memunculkan/menyembunyikan menu kustom.")
for i in range(10):
    st.write(f"Baris Konten {i+1}...")

def render_product_card(row, discount=False):
    """Merender satu kartu produk menggunakan kolom Streamlit."""
    
    image_url = row.get('ImageURL') or "https://via.placeholder.com/150?text=No+Image"
    
    # Gunakan warna sekunder di card border/shadow
    card_style = f"border: 1px solid {COLOR_SECONDARY}; border-radius: 8px; padding: 10px; height: 350px; text-align: center; box-shadow: 2px 2px 5px rgba(0,0,0,0.05);"
    st.markdown(f'<div style="{card_style}">', unsafe_allow_html=True)
    
    # 1. Image
    st.image(image_url, width=150)
    
    # 2. Name (Potong nama agar rapi)
    name_parts = row['Name'].split(' ')
    display_name = " ".join(name_parts[:4]) + ("..." if len(name_parts) > 4 else "")
    st.markdown(f"**{display_name}**", help=row['Name'])
    
    # 3. Price & Discount (Simulasi)
    price = row.get('Price', 50000)
    
    if discount:
        original_price = price * 1.5 
        st.markdown(f"""
            <p style="margin: 0; font-size: 14px; color: grey; text-decoration: line-through;">
                Rp {int(original_price):,}
            </p>
            <p style="margin: 0; font-size: 18px; color: #DC3545; font-weight: bold;">
                Rp {int(price):,} 
            </p>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
            <p style="margin: 0; font-size: 18px; color: black; font-weight: bold;">
                Rp {int(price):,} 
            </p>
        """, unsafe_allow_html=True)


# 4. Footer (Selalu ditampilkan)
render_footer()