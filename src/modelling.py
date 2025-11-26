# src/modelling.py

import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import logging

logger = logging.getLogger(__name__)

def build_hybrid_model(df: pd.DataFrame, tfidf_matrix) -> np.ndarray:
    """Membangun matriks Hybrid Similarity."""
    logger.info("Membangun Hybrid Model (Similarity Matrix)...")
    
    # 1. Content-based similarity (dari TF-IDF)
    content_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

    # 2. Numeric similarity (dari Rating dan ReviewCount scaled log)
    num_features = df[['Rating_scaled', 'ReviewCount_scaled_log']].values
    numeric_sim = cosine_similarity(num_features, num_features)

    # 3. Hybrid similarity (kombinasi 40% Content, 60% Numeric)
    hybrid_sim = 0.4 * content_sim + 0.6 * numeric_sim
    
    logger.info(f"Hybrid Similarity matrix shape: {hybrid_sim.shape}")
    return hybrid_sim

def calculate_evaluation_metrics(df: pd.DataFrame, hybrid_sim: np.ndarray) -> dict:
    """Menghitung rata-rata similarity top-K untuk evaluasi model global."""
    results = []
    n_products = hybrid_sim.shape[0]
    
    # Pastikan ukuran df dan hybrid_sim sama
    if len(df) != n_products:
        df = df.iloc[:n_products].reset_index(drop=True)
        
    for idx in range(n_products):
        # Ambil 5 produk paling mirip (diurutkan [1:6])
        sim_scores = sorted(list(enumerate(hybrid_sim[idx])), key=lambda x: x[1], reverse=True)[1:6]
        if sim_scores:
            avg_sim = np.mean([s[1] for s in sim_scores])
            results.append(avg_sim)
        else:
            results.append(0)
    
    df_similarity_eval = pd.DataFrame(results, columns=['Average_Similarity'])
    
    return {
        "avg_topk_similarity": df_similarity_eval["Average_Similarity"].mean(),
        "global_mean_similarity": df_similarity_eval["Average_Similarity"].mean()
    }