# src/evaluasiLlm.py

import os
import pandas as pd
from pydantic import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
import logging
import streamlit as st

logger = logging.getLogger(__name__)

# --- Setup API Key ---
def load_api_key():
    """Mengambil API key dari Streamlit Secrets atau Environment Variable."""
    
    # 1. Coba ambil dari Streamlit Secrets
    if "GOOGLE_API_KEY" in st.secrets:
        os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
        logger.info("GOOGLE_API_KEY berhasil dimuat dari Streamlit Secrets.")
    
    # 2. Jika tidak ada di Streamlit Secrets, cek environment variable
    elif "GOOGLE_API_KEY" not in os.environ:
        # Jika Anda menjalankan di lingkungan non-Streamlit, ini akan gagal,
        # tetapi di Streamlit Cloud atau lingkungan lokal yang dikonfigurasi, ini aman.
        raise EnvironmentError("GOOGLE_API_KEY tidak ditemukan di Streamlit Secrets maupun Environment Variables.")

# --- Pydantic Schemas ---
class HybridEvaluation(BaseModel):
    score: int = Field(..., description="Skor penilaian 1–10")
    description: str = Field(..., description="Deskripsi singkat tentang kualitas rekomendasi")
    reasons: list[str] = Field(..., description="Alasan penilaian")
    summary: str = Field(..., description="Ringkasan keseluruhan evaluasi hybrid")

class QueryInterpretation(BaseModel):
    product_name: str = Field(..., description="Nama produk atau kata kunci yang paling relevan")
    keywords: list[str] = Field([], description="Keywords relevan")

# --- LLM Tools Class ---
class LLMTools:
    def __init__(self, model_name="gemini-2.5-flash", temperature=0):
        load_api_key()
        self.llm = ChatGoogleGenerativeAI(
            model=model_name,
            temperature=temperature
        )

    def interpret_query_with_llm(self, query: str) -> str:
        """Menginterpretasikan query pengguna menjadi nama produk (untuk 'ADD')."""
        parser_q = PydanticOutputParser(pydantic_object=QueryInterpretation)
        
        prompt_q = ChatPromptTemplate.from_messages([
            ("system", "Kamu adalah assistant yang mengubah query pengguna menjadi nama produk atau kata kunci yang cocok untuk dataset produk."),
            ("human", "User input: {query}\n\nBerikan JSON dengan fields 'product_name' (string) dan 'keywords' (list of strings).\n{format_instructions}")
        ])
        
        chain = prompt_q | self.llm | parser_q
        
        try:
            out = chain.invoke({
                "query": query,
                "format_instructions": parser_q.get_format_instructions()
            })
            return out.product_name
        except Exception as e:
            logger.warning(f"LLM interpret error: {e} — falling back to raw query")
            return query # Fallback
            
    def evaluate_recommendation_with_llm(self, df_rekom: pd.DataFrame) -> HybridEvaluation | None:
        """Mengevaluasi hasil rekomendasi menggunakan LLM."""
        parser_eval = PydanticOutputParser(pydantic_object=HybridEvaluation)
        
        prompt_eval = ChatPromptTemplate.from_messages([
            ("system", "Kamu adalah evaluator untuk sistem rekomendasi hybrid."),
            ("human",
                "Berikut daftar rekomendasi:\n{rekom}\n\nTolong lakukan evaluasi dengan format berikut:\n{format_instructions}")
        ])
        
        chain = prompt_eval | self.llm | parser_eval

        try:
            # Gunakan to_string agar format data lebih mudah dibaca LLM
            rekom_str = df_rekom[['Name','Brand','Category','Rating', 'ReviewCount']].to_string(index=False)
            
            inputs = {
                "rekom": rekom_str,
                "format_instructions": parser_eval.get_format_instructions()
            }

            result = chain.invoke(inputs)
            return result
        
        except Exception as e:
            logger.error(f"LLM evaluation error: {e}")
            return None # Fallback