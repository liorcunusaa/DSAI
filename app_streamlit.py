# app_streamlit.py

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

# --- Konfigurasi Global ---
DATA_FILE_PATH = 'data/product_data.csv'

# --- Modifikasi Fungsi Visualisasi (untuk Streamlit) ---
def display_evaluation_ui(evaluation: HybridEvaluation):
    """Menampilkan hasil evaluasi LLM menggunakan komponen Streamlit."""
    
    score = evaluation.score
    description = evaluation.description
    reasons = evaluation.reasons
    summary = evaluation.summary

    if score >= 8:
        score_color = "green"
    elif score >= 5:
        score_color = "orange"
    else:
        score_color = "red"

    st.markdown("---")
    st.subheader("📊 Hybrid Recommendation Evaluation")
    
    st.markdown(f"**⭐ Score: :{score_color}[{score}/10]**")
    
    st.markdown("**Description:**")
    st.info(description)

    st.markdown("**Reasons:**")
    for r in reasons:
        st.markdown(f"- {r}")

    st.markdown("**Summary:**")
    st.markdown(f"> {summary}")
    st.markdown("---")

# --- Inisialisasi Sistem (Menggunakan Cache Streamlit) ---
@st.cache_resource
def initialize_system():
    """Memuat data, preprocessing, dan membangun model. Dicache agar cepat."""
    try:
        st.info("🚀 Menginisialisasi Sistem Rekomendasi...")
        
        # 1. Load Data
        df = load_local_data(DATA_FILE_PATH)
        if df.empty:
            st.error("Gagal memuat data. Pastikan 'data/product_data.csv' ada.")
            return None, None, None, None

        # 2. Preprocessing & Feature Engineering
        df = clean_and_handle_missing_values(df)
        df, tfidf_matrix = create_features(df)
        
        # 3. Modelling & Metrics
        hybrid_sim = build_hybrid_model(df, tfidf_matrix)
        metrics = calculate_evaluation_metrics(df, hybrid_sim)
        
        # 4. LLM & Recommender Setup
        llm_tools = LLMTools()
        recommender = IntegratedRecommender(df, hybrid_sim)
        
        st.success("✅ Sistem Berhasil Diinisialisasi!")
        return df, recommender, llm_tools, metrics
        
    except EnvironmentError as e:
        st.error(f"❌ Error Inisialisasi LLM: {e}")
        return None, None, None, None
    except Exception as e:
        st.error(f"❌ Error Sistem: {e}")
        return None, None, None, None

# --- Main Streamlit App ---

def main_app():
    st.set_page_config(
        page_title="Hybrid Product Recommender",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    st.title("🛍️ Hybrid Product Recommender")
    st.markdown("Sistem rekomendasi produk berbasis konten, popularitas, dan didukung oleh interpretasi query LLM (Gemini).")

    # Inisialisasi Sistem
    df, recommender, llm_tools, metrics = initialize_system()

    if recommender is None:
        return # Hentikan jika inisialisasi gagal
        
    # --- Sidebar for Metrics & Info ---
    with st.sidebar:
        st.header("⚙️ Konfigurasi & Metrik")
        
        st.metric(label="Jumlah Produk Unik", value=len(df))
        st.metric(label="Rata-rata Top-K Similarity", value=f"{metrics.get('avg_topk_similarity', 0):.4f}")
        
        st.subheader("Opsi Rekomendasi")
        use_llm = st.checkbox("Gunakan LLM Query Interpreter (Gemini)?", value=True)
        top_n = st.slider("Jumlah Rekomendasi (Top N)", min_value=1, max_value=20, value=5)

    # --- Main Input Area ---
    st.header("1. Cari Produk")
    
    product_query = st.text_input(
        "Masukkan nama produk atau deskripsi yang Anda cari:", 
        placeholder="Contoh: 'Foundation tahan lama untuk kulit berminyak'"
    )
    
    # --- Recommendation Logic ---
    if st.button("Run Recommendation", type="primary") and product_query:
        
        with st.spinner(f"Mencari rekomendasi untuk '{product_query}'..."):
            
            # 1. Interpretasi Query (Jika diaktifkan)
            interpreted = product_query
            if use_llm and llm_tools:
                st.info(f"Input User: **{product_query}**")
                interpreted = llm_tools.interpret_query_with_llm(product_query)
                st.success(f"Diterjemahkan (LLM): **{interpreted}**")

            # 2. Ambil Rekomendasi
            recs = recommender.get_recommendations(interpreted, top_n)

            if isinstance(recs, str): # Error/Not Found
                st.error(recs)
                st.session_state['current_rekom'] = None
            elif recs.empty:
                st.warning("⚠️ Tidak ada rekomendasi ditemukan.")
                st.session_state['current_rekom'] = None
            else:
                st.subheader(f"2. Hasil Rekomendasi (Top {len(recs)})")
                st.dataframe(recs[['Name', 'Brand', 'Category', 'Rating', 'ReviewCount', 'final_score']], use_container_width=True)
                
                st.session_state['current_rekom'] = recs.copy()
                st.metric(label="Rata-rata Final Score", value=f"{recs['final_score'].mean():.4f}")

    # --- LLM Evaluation Section (Hanya jika ada rekomendasi) ---
    if 'current_rekom' in st.session_state and st.session_state['current_rekom'] is not None:
        
        st.header("3. Evaluasi Kualitas Rekomendasi dengan LLM")
        
        if st.button("Evaluate with LLM (Gemini)", key="btn_eval"):
            if not llm_tools:
                st.error("LLM Tools tidak tersedia. Cek API Key Anda.")
            else:
                with st.spinner("⏳ Mengirim hasil ke Gemini untuk dievaluasi..."):
                    eval_res = llm_tools.evaluate_recommendation_with_llm(st.session_state['current_rekom'])
                    
                    if eval_res:
                        display_evaluation_ui(eval_res)
                    else:
                        st.error("❌ Gagal mendapatkan hasil evaluasi dari LLM.")
    
    # --- EDA and Visualizations ---
    st.header("4. Analisis Data dan Visualisasi Model")
    if st.button("Tampilkan EDA & Visualisasi", key="btn_viz"):
        
        st.subheader("Exploratory Data Analysis (EDA)")
        
        # 1. Distribusi Rating
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.histplot(df['Rating'], bins=20, kde=True, ax=ax)
        ax.set_title("Distribusi Rating Produk")
        st.pyplot(fig)
        
        # 2. Top 10 Kategori
        fig, ax = plt.subplots(figsize=(8, 4))
        df['Category'].value_counts().head(10).plot(kind='barh', color='skyblue', ax=ax)
        ax.set_title("Top 10 Product Categories")
        st.pyplot(fig)
        
        st.subheader("Visualisasi Model: Korelasi Kemiripan Produk")
        
        # 3. Heatmap
        # Ambil 10 produk pertama untuk visualisasi
        n_viz = min(10, len(df))
        sample_indices = range(n_viz)
        sample_names = df['Name'].iloc[sample_indices].str[:30]
        
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(
            recommender.hybrid_sim[np.ix_(sample_indices, sample_indices)],
            xticklabels=sample_names,
            yticklabels=sample_names,
            cmap='YlGnBu',
            ax=ax
        )
        ax.set_title(f"Korelasi Kemiripan Antar {n_viz} Produk (Hybrid Similarity)")
        plt.xticks(rotation=90)
        plt.yticks(rotation=0)
        st.pyplot(fig)


if __name__ == '__main__':
    main_app()