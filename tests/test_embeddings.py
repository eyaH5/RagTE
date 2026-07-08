from __future__ import annotations

from api import embeddings as embeddings_module


class _FakeResponse:
    def __init__(self, status_code: int, payload):
        self.status_code = status_code
        self._payload = payload

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


class _FakeClient:
    def __init__(self, responses):
        self._responses = list(responses)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def post(self, url, json):
        return self._responses.pop(0)


def test_tei_embed_flattens_single_string_response(monkeypatch):
    monkeypatch.setattr(
        embeddings_module.httpx,
        "Client",
        lambda timeout=120.0: _FakeClient([_FakeResponse(200, [[1.0, 2.0, 3.0]])]),
    )

    embedder = embeddings_module.Embedder("BAAI/bge-m3", "http://localhost:8002")

    assert embedder._encode_via_tei("bonjour") == [1.0, 2.0, 3.0]


def test_tei_embed_keeps_batch_shape_for_list_inputs(monkeypatch):
    monkeypatch.setattr(
        embeddings_module.httpx,
        "Client",
        lambda timeout=120.0: _FakeClient([_FakeResponse(200, [[1.0, 2.0], [3.0, 4.0]])]),
    )

    embedder = embeddings_module.Embedder("BAAI/bge-m3", "http://localhost:8002")

    assert embedder._encode_via_tei(["bonjour", "salut"]) == [[1.0, 2.0], [3.0, 4.0]]
