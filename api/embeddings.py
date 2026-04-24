from sentence_transformers import SentenceTransformer
import torch
from functools import lru_cache
from api.config import get_settings

settings = get_settings()

@lru_cache()
def get_embedder():
    """
    Returns the SentenceTransformer embedding model.
    Loads onto CUDA if available.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return SentenceTransformer("BAAI/bge-m3").to(device)
