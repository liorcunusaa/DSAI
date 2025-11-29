# app_streamlit.py

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import time

# --- Mock Modules (JIKA TIDAK ADA MODULE LOKAL) ---
# Hapus bagian "TRY/EXCEPT" ini jika file src/ kamu sudah lengkap.
# Ini hanya agar kode bisa jalan di saya tanpa file src aslimu.
try:
    from src.data_loader import load_local_data
    from src.preprocessing import clean_and_handle_missing_values
    from src.feature_engineering import create_features
    from src.modelling import build_hybrid_model, calculate_evaluation_metrics
    from src.integratedRecommender import IntegratedRecommender
    from src.evaluasiLlm import LLMTools, HybridEvaluation
except ImportError:
    # Dummy classes placeholders agar tidak error saat copy-paste
    class HybridEvaluation:
        def __init__(self): self.score=8.5; self.description="Good"; self.reasons=["Relevant"]; self.summary="Ok"
    class LLMTools:
        def interpret_query_with_llm(self, q): return q
        def evaluate_recommendation_with_llm(self, recs): return HybridEvaluation()
    def load_local_data(path): 
        # Membuat dummy data jika file tidak ada
        data = {
            'Name': [f'Product {i}' for i in range(20)],
            'Brand': ['Brand A', 'Brand B']*10,
            'Category': ['Skincare', 'Gadget', 'Fashion', 'Home']*5,
            'Rating': np.random.randint(3, 6, 20),
            'ReviewCount': np.random.randint(10, 100, 20),
            'Description': ['Deskripsi produk contoh yang cukup panjang untuk demo layout UI.']*20,
            'ImageURL': ['https://via.placeholder.com/300']*20
        }
        return pd.DataFrame(data)
    def clean_and_handle_missing_values(df): return df
    def create_features(df): return df, None
    def build_hybrid_model(df, tf): return np.eye(len(df))
    def calculate_evaluation_metrics(df, sim): return {}
    class IntegratedRecommender:
        def __init__(self, df, sim): self.df=df; self.hybrid_sim=sim
        def get_recommendations(self, q, n): 
            df = self.df.copy()
            df['final_score'] = np.random.rand(len(df))
            return df.head(n)

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Konfigurasi Global ---
DATA_FILE_PATH = 'data/product_data.csv'

# --- Custom CSS (The "Wow" Factor) ---
def inject_custom_css():
    st.markdown("""
        <style>
        /* 1. GLOBAL FONT & BACKGROUND */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;800&display=swap');
        
        html, body, [class*="css"] {
            font-family: 'Inter', sans-serif;
        }
        
        /* Background Utama: Light Gray agar Card Putih Pop-up */
        .stApp {
            background-color: #f8f9fa;
        }

        /* 2. HEADER CUSTOMIZATION */
        h1, h2, h3 {
            color: #1E293B !important; /* Slate 800 */
            font-weight: 800 !important;
        }
        
        /* 3. CARD STYLE (Container) */
        /* Menargetkan container border agar terlihat seperti kartu modern */
        [data-testid="stVerticalBlockBorderWrapper"] {
            background-color: #FFFFFF;
            border: 1px solid #e2e8f0;
            border-radius: 12px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.02);
            padding: 1rem;
            transition: all 0.3s ease;
        }
        
        /* Efek Hover pada Card */
        [data-testid="stVerticalBlockBorderWrapper"]:hover {
            box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05);
            transform: translateY(-2px);
            border-color: #cbd5e1;
        }

        /* Khusus Container Header (Biru) - Override style di atas */
        [data-testid="stVerticalBlockBorderWrapper"]:has(#header-marker) {
            background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%); /* Dark Blue Gradient */
            color: white;
            border: none;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        }
        
        /* Text di dalam Header harus Putih */
        [data-testid="stVerticalBlockBorderWrapper"]:has(#header-marker) h1,
        [data-testid="stVerticalBlockBorderWrapper"]:has(#header-marker) span,
        [data-testid="stVerticalBlockBorderWrapper"]:has(#header-marker) div {
            color: white !important;
        }

        /* 4. BUTTON STYLING */
        /* Primary Button (Gradient) */
        div.stButton > button[kind="primary"] {
            background: linear-gradient(to right, #3b82f6, #2563eb);
            color: white;
            border: none;
            border-radius: 8px;
            font-weight: 600;
            transition: all 0.2s;
        }
        div.stButton > button[kind="primary"]:hover {
            box-shadow: 0 4px 12px rgba(37, 99, 235, 0.3);
            transform: scale(1.02);
        }

        /* Secondary Button (Outline) */
        div.stButton > button[kind="secondary"] {
            background-color: transparent;
            border: 1px solid #cbd5e1;
            color: #475569;
            border-radius: 8px;
        }
        div.stButton > button[kind="secondary"]:hover {
            background-color: #f1f5f9;
            border-color: #94a3b8;
            color: #1e293b;
        }

        /* 5. INPUT FIELDS */
        .stTextInput > div > div > input {
            border-radius: 8px;
            border: 1px solid #e2e8f0;
            padding: 10px 12px;
        }
        .stTextInput > div > div > input:focus {
            border-color: #3b82f6;
            box-shadow: 0 0 0 2px rgba(59, 130, 246, 0.2);
        }

        /* 6. UTILITIES */
        .product-title {
            font-size: 0.95rem;
            font-weight: 700;
            color: #1e293b;
            line-height: 1.4;
            height: 2.8em; /* Batasi 2 baris */
            overflow: hidden;
            display: -webkit-box;
            -webkit-line-clamp: 2;
            -webkit-box-orient: vertical;
            margin-bottom: 0.5rem;
        }
        
        .product-price {
            font-size: 0.9rem;
            color: #64748b;
            margin-bottom: 0.5rem;
        }

        .metric-badge {
            background-color: #dbeafe;
            color: #1d4ed8;
            padding: 2px 8px;
            border-radius: 12px;
            font-size: 0.75rem;
            font-weight: 600;
        }
        </style>
    """, unsafe_allow_html=True)

# --- Fungsi Bantu UI & Logika ---

def display_evaluation_ui(evaluation: HybridEvaluation):
    """Menampilkan hasil evaluasi LLM dengan gaya modern."""
    score = evaluation.score
    
    # Menentukan warna berdasarkan skor
    if score >= 8:
        color = "#22c55e" # Green
        bg_color = "#dcfce7"
        icon = "🌟 Excellent"
    elif score >= 5:
        color = "#f97316" # Orange
        bg_color = "#ffedd5"
        icon = "⚖️ Moderate"
    else:
        color = "#ef4444" # Red
        bg_color = "#fee2e2"
        icon = "⚠️ Low Relevance"

    st.markdown("---")
    
    with st.container(border=True):
        c1, c2 = st.columns([1, 4])
        with c1:
            st.markdown(f"""
                <div style="background-color: {bg_color}; border-radius: 12px; padding: 20px; text-align: center;">
                    <h1 style="color: {color} !important; margin: 0; font-size: 3rem;">{score}</h1>
                    <p style="color: {color}; font-weight: bold; margin: 0;">{icon}</p>
                </div>
            """, unsafe_allow_html=True)
        with c2:
            st.subheader("Analisis AI")
            st.info(evaluation.description, icon="🤖")
            
            with st.expander("Lihat Alasan Detail"):
                for r in evaluation.reasons:
                    st.markdown(f"- {r}")
                st.markdown(f"**Kesimpulan:** _{evaluation.summary}_")

# --- POP-UP DETAIL PRODUK (Dialog) ---
@st.dialog("Detail Produk", width="large")
def show_product_popup(product_data, score=None):
    # Layout Grid: Gambar di Kiri, Info di Kanan
    c_img, c_info = st.columns([1, 1.2], gap="medium")
    
    with c_img:
        img_url = str(product_data.get('ImageURL', '')).split('|')[0]
        if img_url and img_url != 'nan':
            st.image(img_url, use_container_width=True)
        else:
            st.markdown('<div style="height:300px; background:#f1f5f9; border-radius:10px; display:flex; align-items:center; justify-content:center; color:#94a3b8;">No Image</div>', unsafe_allow_html=True)

    with c_info:
        st.caption(product_data.get('Brand', 'Generic Brand'))
        st.markdown(f"## {product_data.get('Name', 'No Name')}")
        
        # Rating & Reviews
        r_val = int(product_data.get('Rating', 0))
        stars = "★" * r_val + "☆" * (5 - r_val)
        st.markdown(f"<span style='color:#f59e0b; font-size:1.2rem;'>{stars}</span> <span style='color:#64748b; font-size:0.9rem;'>({product_data.get('ReviewCount', 0)} reviews)</span>", unsafe_allow_html=True)
        
        if score:
            st.markdown(f"<span class='metric-badge'>Match Score: {score:.2f}</span>", unsafe_allow_html=True)
            
        st.markdown("---")
        st.write(product_data.get('Description', 'Tidak ada deskripsi.'))
        
        st.markdown("<br>", unsafe_allow_html=True)
        col_b1, col_b2 = st.columns(2)
        with col_b1:
            if st.button("Add to Cart 🛒", use_container_width=True):
                st.toast("Added to cart!", icon="✅")
        with col_b2:
            st.button("Buy Now ⚡", type="primary", use_container_width=True)

    st.markdown("---")
    st.markdown("### Spesifikasi Teknis")
    
    # Menampilkan atribut lain dalam bentuk tabel rapi
    exclude = ['Name', 'Brand', 'Category', 'Rating', 'ReviewCount', 'Description', 'ImageURL', 'final_score', 'Name_norm']
    spec_data = {k: v for k, v in product_data.items() if k not in exclude and pd.notna(v)}
    
    if spec_data:
        # Tampilkan sebagai grid kecil
        cols = st.columns(3)
        for i, (k, v) in enumerate(spec_data.items()):
            with cols[i % 3]:
                st.markdown(f"**{k}**")
                st.caption(str(v))

def render_product_card(row, full_df=None):
    """Merender kartu produk yang bersih."""
    # Gunakan container border=True untuk card effect (di-style via CSS)
    with st.container(border=True):
        # 1. Image Area (Fixed Aspect Ratio)
        img_url = str(row['ImageURL']).split('|')[0] if pd.notna(row['ImageURL']) else None
        
        if img_url:
            st.markdown(f"""
                <div style="height: 160px; display: flex; justify-content: center; align-items: center; overflow: hidden; margin-bottom: 12px; border-radius: 8px;">
                    <img src="{img_url}" style="height: 100%; width: 100%; object-fit: contain;">
                </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
                <div style="height: 160px; background-color: #f1f5f9; display: flex; justify-content: center; align-items: center; margin-bottom: 12px; border-radius: 8px; color: #94a3b8;">
                    📷 No Image
                </div>
            """, unsafe_allow_html=True)

        # 2. Content
        st.markdown(f"<div class='product-title' title='{row['Name']}'>{row['Name']}</div>", unsafe_allow_html=True)
        
        # Brand & Rating
        c_brand, c_rating = st.columns([2, 1])
        with c_brand:
            st.caption(row.get('Brand', 'Generic'))
        with c_rating:
            st.markdown(f"⭐ **{int(row.get('Rating', 0))}**")
            
        # Score Badge jika hasil rekomendasi
        if 'final_score' in row:
            st.markdown(f"<div style='margin-top:4px;'><span class='metric-badge'>Score: {row['final_score']:.2f}</span></div>", unsafe_allow_html=True)
        
        st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
        
        # 3. Action Button
        if st.button("Detail", key=f"btn_{row.name}", use_container_width=True):
            full_data = full_df.loc[row.name] if full_df is not None else row
            show_product_popup(full_data, score=row.get('final_score'))

@st.cache_resource
def initialize_system():
    """Load data & models."""
    try:
        df = load_local_data(DATA_FILE_PATH)
        if df.empty: return None, None, None, None
        df = clean_and_handle_missing_values(df)
        df, tfidf = create_features(df)
        hybrid_sim = build_hybrid_model(df, tfidf)
        metrics = calculate_evaluation_metrics(df, hybrid_sim)
        llm_tools = LLMTools()
        recommender = IntegratedRecommender(df, hybrid_sim)
        return df, recommender, llm_tools, metrics
    except Exception as e:
        logger.error(f"Init Error: {e}")
        return None, None, None, None

# --- PAGES ---

def page_recommender(df, recommender, llm_tools):
    # --- HEADER SECTION (Dark Blue Gradient via CSS) ---
    with st.container(border=True):
        st.markdown('<span id="header-marker"></span>', unsafe_allow_html=True) # CSS Target
        
        c1, c2, c3, c4 = st.columns([0.5, 4, 1.5, 1], gap="small")
        
        with c1:
             if st.button("⬅️", help="Back to Home"):
                 st.session_state["current_page"] = "home"
                 st.rerun()
        
        with c2:
            default_q = st.session_state.get("global_search_query", "")
            query = st.text_input("Search", value=default_q, placeholder="Cari: 'Laptop gaming murah'...", label_visibility="collapsed")
            
        with c3:
            # Layout tombol aksi dalam kolom
            cb1, cb2 = st.columns(2)
            with cb1:
                do_search = st.button("🔍 Cari", type="primary", use_container_width=True)
            with cb2:
                can_eval = 'current_rekom' in st.session_state and st.session_state['current_rekom'] is not None
                do_eval = st.button("✨ AI Eval", disabled=not can_eval, use_container_width=True)
        
        with c4:
            top_n = st.number_input("Jml", 5, 50, 10, 5, label_visibility="collapsed")

    # --- LOGIKA SEARCH ---
    trigger = st.session_state.get("trigger_search", False)
    
    if (do_search or trigger) and query:
        st.session_state.trigger_search = False
        with st.spinner("🔍 Menganalisis preferensi Anda..."):
            # Simulasi interpretasi LLM
            interpreted_q = llm_tools.interpret_query_with_llm(query)
            if interpreted_q != query:
                st.toast(f"AI: Saya perjelas pencarianmu menjadi '{interpreted_q}'", icon="🤖")
            
            recs = recommender.get_recommendations(interpreted_q, int(top_n))
            
            if not isinstance(recs, str) and not recs.empty:
                st.session_state['current_rekom'] = recs
                if 'last_eval_result' in st.session_state: del st.session_state['last_eval_result']
            else:
                st.error("Produk tidak ditemukan.")
        st.rerun()

    # --- LOGIKA EVALUASI ---
    if do_eval and 'current_rekom' in st.session_state:
        with st.spinner("🤖 Gemini sedang membaca hasil pencarian..."):
            res = llm_tools.evaluate_recommendation_with_llm(st.session_state['current_rekom'])
            st.session_state['last_eval_result'] = res

    # --- MAIN CONTENT ---
    st.markdown("<br>", unsafe_allow_html=True)
    
    # 1. Tampilkan Evaluasi jika ada
    if 'last_eval_result' in st.session_state:
        display_evaluation_ui(st.session_state['last_eval_result'])
    
    # 2. Grid Produk
    if 'current_rekom' in st.session_state:
        recs = st.session_state['current_rekom']
        st.subheader(f"Hasil: {len(recs)} Produk Relevan")
        
        cols = st.columns(5) # Responsive Grid 5 Kolom
        for idx, (index, row) in enumerate(recs.iterrows()):
            with cols[idx % 5]:
                render_product_card(row, full_df=df)

    # 3. Footer Stats (EDA Mini)
    st.markdown("---")
    with st.expander("📊 Statistik Data Produk"):
        ec1, ec2 = st.columns(2)
        with ec1:
            st.caption("Distribusi Rating")
            fig, ax = plt.subplots(figsize=(5,2))
            sns.histplot(df['Rating'], bins=5, ax=ax, color="#3b82f6")
            ax.set_frame_on(False)
            st.pyplot(fig)
        with ec2:
            st.caption("Top Kategori")
            st.bar_chart(df['Category'].value_counts().head(5), color="#3b82f6")

def page_home(df):
    # --- HERO SECTION ---
    # Menggunakan Container dengan style gradient khusus
    with st.container(border=True):
        st.markdown('<span id="header-marker"></span>', unsafe_allow_html=True) # Menggunakan style header biru
        
        c_hero_txt, c_hero_img = st.columns([2, 1])
        
        with c_hero_txt:
            st.markdown("# 👋 Hi, Mau cari apa hari ini?")
            st.markdown("Temukan produk terbaik dengan bantuan **AI Recommendation Engine** kami.")
            
            # Search Bar Besar
            q = st.text_input("Search bar", placeholder="Ketik kata kunci (misal: 'Sepatu lari warna biru')...", label_visibility="collapsed", key="home_search")
            
            if st.button("Mulai Pencarian 🚀", type="primary"):
                if q:
                    st.session_state["global_search_query"] = q
                    st.session_state["trigger_search"] = True
                    st.session_state["current_page"] = "recommender"
                    st.rerun()
                else:
                    st.toast("Ketik sesuatu dulu ya!", icon="😅")
        
        with c_hero_img:
            # Placeholder illustrasi
            st.markdown("""
            <div style="display:flex; justify-content:center; align-items:center; height:100%;">
                <span style="font-size: 8rem;">🛍️</span>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # --- FEATURED PRODUCTS ---
    st.markdown("### 🔥 Trending Now")
    
    # Ambil 10 produk acak sebagai display awal
    if not df.empty:
        display_df = df.sample(min(10, len(df)))
        
        cols = st.columns(5)
        for idx, (index, row) in enumerate(display_df.iterrows()):
            with cols[idx % 5]:
                render_product_card(row, full_df=df)
    
    st.markdown("<br><hr>", unsafe_allow_html=True)
    st.markdown("<div style='text-align:center; color:#94a3b8;'>© 2025 5uper Market AI • Powered by Streamlit & Gemini</div>", unsafe_allow_html=True)


# --- MAIN APP CONTROLLER ---

def main():
    st.set_page_config(page_title="5uper Market AI", page_icon="🛒", layout="wide")
    
    # 1. Inject Style
    inject_custom_css()
    
    # 2. Init State
    if "current_page" not in st.session_state: st.session_state["current_page"] = "home"
    
    # 3. Load System
    df, recommender, llm_tools, metrics = initialize_system()
    
    if df is None:
        st.error("Gagal memuat data. Pastikan file 'data/product_data.csv' tersedia.")
        return

    # 4. Routing
    if st.session_state["current_page"] == "home":
        page_home(df)
    elif st.session_state["current_page"] == "recommender":
        page_recommender(df, recommender, llm_tools)

if __name__ == "__main__":
    main()