# src/preprocessing.py

import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

def clean_and_handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Melakukan pembersihan dan penanganan missing value."""
    df = df.copy()
    logger.info("Memulai Pre-Processing dan Data Cleaning...")
    
    # 1. Handling Missing Value (dengan nilai default/rata-rata)
    df['Rating'] = df['Rating'].fillna(df['Rating'].mean())
    df['ReviewCount'] = df['ReviewCount'].fillna(0)
    df['Description'] = df['Description'].fillna('')
    df['Tags'] = df['Tags'].fillna('')
    df['Name'] = df['Name'].fillna('')
    df['Category'] = df['Category'].fillna('Unknown')
    df['Brand'] = df['Brand'].fillna('Unknown')

    # 2. Hapus duplikat berdasarkan ProdID
    if 'ProdID' in df.columns:
        df = df.drop_duplicates(subset=['ProdID']).reset_index(drop=True)
    else:
        logger.warning("Kolom 'ProdID' tidak ditemukan. Duplikat tidak dihapus.")
        df = df.reset_index(drop=True)

    # 3. Pembersihan Rating Lanjutan
    df['Rating'] = df['Rating'].replace(0, np.nan)
    # Isi NaN dengan rata-rata rating per Brand
    df['Rating'] = df.groupby('Brand')['Rating'].transform(lambda x: x.fillna(x.mean()))
    # Isi NaN yang tersisa dengan rata-rata global
    df['Rating'] = df['Rating'].fillna(df['Rating'].mean())

    logger.info(f"Pembersihan selesai. Ukuran data: {df.shape}")
    return df