import numpy as np
import pytest
from src.segmentation.embeddings import (
    get_embeddings,
    cosine_similarity,
    compute_coherence_scores,
    segment_coherence,
)

RELATED_SENTENCES = [
    "Large language models require significant computational resources.",
    "Inference costs grow quadratically with sequence length.",
    "Techniques like quantization help reduce inference costs.",
]

UNRELATED_SENTENCES = [
    "Large language models require significant computational resources.",
    "The weather today is sunny and warm.",
    "I enjoy eating pizza on weekends.",
]


def test_get_embeddings_returns_numpy_array():
    embeddings = get_embeddings(RELATED_SENTENCES)
    assert isinstance(embeddings, np.ndarray)


def test_get_embeddings_shape():
    embeddings = get_embeddings(RELATED_SENTENCES)
    assert embeddings.shape[0] == len(RELATED_SENTENCES)
    assert embeddings.shape[1] > 0


def test_cosine_similarity_identical_vectors():
    v = np.array([1.0, 0.5, 0.3])
    assert cosine_similarity(v, v) == pytest.approx(1.0, abs=1e-6)


def test_cosine_similarity_range():
    embeddings = get_embeddings(RELATED_SENTENCES)
    score = cosine_similarity(embeddings[0], embeddings[1])
    assert -1.0 <= score <= 1.0


def test_related_sentences_higher_coherence_than_unrelated():
    related_score = segment_coherence(RELATED_SENTENCES)
    unrelated_score = segment_coherence(UNRELATED_SENTENCES)
    assert related_score > unrelated_score


def test_segment_coherence_single_sentence():
    score = segment_coherence(["Only one sentence here."])
    assert score == 1.0


def test_compute_coherence_scores_length():
    scores = compute_coherence_scores(RELATED_SENTENCES)
    assert len(scores) == len(RELATED_SENTENCES) - 1


def test_segment_coherence_returns_float():
    score = segment_coherence(RELATED_SENTENCES)
    assert isinstance(score, float)
    assert 0.0 <= score <= 1.0