from functools import lru_cache
from typing import Any

import httpx
import torch
from loguru import logger
from sentence_transformers import SentenceTransformer

from api.config import get_settings

settings = get_settings()


def to_builtin_list(value: Any):
    return value.tolist() if hasattr(value, "tolist") else value


class Embedder:
    def __init__(self, model_name: str, tei_url: str | None = None):
        self.model_name = model_name
        self.tei_url = tei_url.rstrip("/") if tei_url else None
        self._local_model: SentenceTransformer | None = None
        self._tei_warned = False

    def encode(self, texts: str | list[str], **kwargs):
        if self.tei_url:
            try:
                return self._encode_via_tei(texts)
            except Exception as exc:
                if not self._tei_warned:
                    logger.warning(
                        "TEI embedding request failed at {} for model {}. Falling back to local SentenceTransformer. Error: {}",
                        self.tei_url,
                        self.model_name,
                        exc,
                    )
                    self._tei_warned = True

        return self._encode_locally(texts, **kwargs)

    def _encode_via_tei(self, texts: str | list[str]):
        with httpx.Client(timeout=120.0) as client:
            embed_response = client.post(
                f"{self.tei_url}/embed",
                json={"inputs": texts, "truncate": True},
            )

            if embed_response.status_code == 404:
                openai_response = client.post(
                    f"{self.tei_url}/v1/embeddings",
                    json={"input": texts, "model": self.model_name},
                )
                openai_response.raise_for_status()
                data = openai_response.json()["data"]
                embeddings = [row["embedding"] for row in data]
                return embeddings[0] if isinstance(texts, str) else embeddings

            embed_response.raise_for_status()
            return embed_response.json()

    def _encode_locally(self, texts: str | list[str], **kwargs):
        if self._local_model is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info("Loading local embedding model {} on {}", self.model_name, device)
            self._local_model = SentenceTransformer(self.model_name).to(device)

        return self._local_model.encode(texts, **kwargs)


@lru_cache()
def get_embedder() -> Embedder:
    return Embedder(settings.EMBEDDING_MODEL, settings.TEI_URL)
