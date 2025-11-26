# src/integratedRecommender.py

import pandas as pd
import numpy as np
from difflib import get_close_matches
import logging

logger = logging.getLogger(__name__)

class IntegratedRecommender:
    def __init__(self, df: pd.DataFrame, hybrid_sim: np.ndarray):
        self.df = df
        self.hybrid_sim = hybrid_sim

    def get_recommendations(self, product_name: str, n: int = 5):
        """Fungsi rekomendasi hybrid utama (digunakan dalam UI/CLI)."""
        product_name = product_name.strip().lower()
        self.df['Name_norm'] = self.df['Name'].str.strip().str.lower()
        
        # 1. Cari produk acuan (termasuk fuzzy match)
        idx = None
        if product_name in self.df['Name_norm'].values:
            idx = self.df[self.df['Name_norm'] == product_name].index[0]
        else:
            # Cari yang mengandung kata kunci (partial match)
            matches = self.df[self.df['Name_norm'].str.contains(product_name, case=False, na=False)]
            if len(matches) > 0:
                idx = matches.index[0]
                logger.info(f"🔍 Produk mirip ditemukan (Partial Match): {self.df.iloc[idx]['Name']}")
            else:
                # Cari yang paling mirip (close match)
                closest = get_close_matches(product_name, self.df['Name_norm'], n=1, cutoff=0.4)
                if closest:
                    idx = self.df[self.df['Name_norm'] == closest[0]].index[0]
                    logger.info(f"🔍 Produk tidak ditemukan persis. Menampilkan hasil mirip (Fuzzy Match): {self.df.iloc[idx]['Name']}")
                else:
                    return f"❌ Produk '{product_name}' tidak ditemukan di dataset."

        # 2. Ambil skor similarity
        sim_scores = list(enumerate(self.hybrid_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:n+20] # ambil lebih banyak kandidat

        # 3. Hitung Final Score
        results = self.df.iloc[[i[0] for i in sim_scores]].copy()
        results['similarity'] = [s[1] for s in sim_scores]

        # Normalisasi ulang rating & review agar 0-1 (untuk final score)
        min_rating, max_rating = self.df['Rating'].min(), self.df['Rating'].max()
        min_review, max_review = self.df['ReviewCount'].min(), self.df['ReviewCount'].max()

        results['rating_norm'] = (results['Rating'] - min_rating) / (max_rating - min_rating)
        results['review_norm'] = (results['ReviewCount'] - min_review) / (max_review - min_review)

        # Final Score: 40% Similarity + 30% Rating + 30% Review
        results['final_score'] = (
            0.4 * results['similarity'] +
            0.3 * results['rating_norm'] +
            0.3 * results['review_norm']
        )

        # 4. Urutkan dan ambil top-n
        results = results.sort_values('final_score', ascending=False)
        recommended = results[['Name','Brand','Category','Rating','ReviewCount','final_score','Description']].head(n)
        
        return recommended