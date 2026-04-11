from sentence_transformers import SentenceTransformer
import numpy as np

_model = None
DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(DEFAULT_EMBEDDING_MODEL)
    return _model


def get_embeddings(texts: list[str]) -> np.ndarray:
    model = _get_model()
    return model.encode(texts, convert_to_numpy=True)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def compute_coherence_scores(sentences: list[str]) -> list[float]:
    if len(sentences) < 2:
        return [1.0]

    embeddings = get_embeddings(sentences)
    scores = []
    for i in range(len(embeddings) - 1):
        score = cosine_similarity(embeddings[i], embeddings[i + 1])
        scores.append(score)
    return scores


def segment_coherence(sentences: list[str]) -> float:
    scores = compute_coherence_scores(sentences)
    return float(np.mean(scores))