# src/visualisasi.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging

logger = logging.getLogger(__name__)

def run_eda(df: pd.DataFrame):
    """Menampilkan plot EDA dasar."""
    logger.info("Menjalankan Exploratory Data Analysis (EDA)...")
    
    # 1. Distribusi Rating Produk
    plt.figure(figsize=(6,4))
    sns.histplot(df['Rating'], bins=20, kde=True)
    plt.title("Distribusi Rating Produk")
    plt.show()

    # 2. Top 10 Kategori
    plt.figure(figsize=(8,4))
    df['Category'].value_counts().head(10).plot(kind='barh', color='skyblue')
    plt.title("Top 10 Product Categories")
    plt.show()

    # 3. Korelasi Antar Nilai Numerik
    plt.figure(figsize=(5,3))
    sns.heatmap(df[['Rating','ReviewCount']].corr(), annot=True, cmap='Blues')
    plt.title("Korelasi antar variabel numerik")
    plt.show()
    logger.info("EDA selesai.")

def plot_hybrid_similarity_heatmap(df: pd.DataFrame, hybrid_sim: np.ndarray, n: int = 10):
    """Membuat Heatmap Korelasi Kemiripan Antar Produk."""
    if len(df) < n:
        n = len(df)
    
    sample_indices = range(n)
    sample_names = df['Name'].iloc[sample_indices].str[:30] # Potong nama agar tidak terlalu panjang

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        hybrid_sim[np.ix_(sample_indices, sample_indices)],
        xticklabels=sample_names,
        yticklabels=sample_names,
        cmap='YlGnBu'
    )
    plt.title(f"Korelasi Kemiripan Antar {n} Produk (Hybrid Similarity)")
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.show()