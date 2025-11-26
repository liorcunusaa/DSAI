# src/feature_engineering.py

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler
import logging

logger = logging.getLogger(__name__)

def create_features(df: pd.DataFrame):
    """Membuat fitur teks (TF-IDF) dan numerik (Scaled) dari DataFrame."""
    df = df.copy()
    logger.info("Memulai Feature Engineering...")
    
    # 1. Gabungkan semua kolom teks
    df['text_features'] = (
        df['Name'] + ' ' +
        df['Description'] + ' ' +
        df['Tags'] + ' ' +
        df['Brand'] + ' ' +
        df['Category']
    )

    # 2. TF-IDF vectorization
    vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
    tfidf_matrix = vectorizer.fit_transform(df['text_features'])
    logger.info(f"TF-IDF matrix shape: {tfidf_matrix.shape}")

    # 3. Normalisasi nilai numerik (Rating & Review Count)
    scaler = MinMaxScaler()
    df[['Rating_scaled', 'ReviewCount_scaled']] = scaler.fit_transform(
        df[['Rating', 'ReviewCount']]
    )

    # Scaling log ReviewCount (untuk model)
    df['review_log'] = np.log1p(df['ReviewCount'])
    df['ReviewCount_scaled_log'] = df['review_log'] / df['review_log'].max()
    
    logger.info("Feature Engineering selesai.")
    return df, tfidf_matrix