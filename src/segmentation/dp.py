# dp.py — algoritmo DP con función de costo compuesta
from src.segmentation.models import Segment, SegmentationResult
from src.segmentation.tokenizer import count_tokens_batch
from src.segmentation.splitter import split_sentences
from src.segmentation.embeddings import get_embeddings, cosine_similarity
import numpy as np

INF = float("inf")
DEFAULT_LAMBDA = 0.5        # peso de coherencia semántica
DEFAULT_FIXED_COST = 1000.0  # overhead fijo por llamada al LLM


def _compute_cost(
    token_count: int,
    embeddings: np.ndarray,
    start: int,
    end: int,
    coherence_lambda: float,
    fixed_cost: float = DEFAULT_FIXED_COST,
) -> float:
    # Componente 1: costo cuadrático de autoatención
    computational_cost = float(token_count ** 2)

    # Componente 2: penalización de coherencia semántica
    if end - start < 2:
        coherence_penalty = 0.0
    else:
        scores = []
        for i in range(start, end - 1):
            score = cosine_similarity(embeddings[i], embeddings[i + 1])
            scores.append(score)
        avg_coherence = float(np.mean(scores))
        coherence_penalty = (1.0 - avg_coherence)

    return computational_cost + coherence_lambda * coherence_penalty + fixed_cost


# prefix sum — permite calcular tokens(i,j) en O(1)
def _build_cumulative_tokens(token_lens: list[int]) -> list[int]:
    cumulative = [0]
    for t in token_lens:
        cumulative.append(cumulative[-1] + t)
    return cumulative


def segment_dp(
    text: str,
    lmin: int = 50,
    lmax: int = 200,
    model: str = "gpt-4o",
    coherence_lambda: float = DEFAULT_LAMBDA,
    fixed_cost: float = DEFAULT_FIXED_COST,
) -> SegmentationResult:
    sentences = split_sentences(text)
    if not sentences:
        raise ValueError("No sentences found in input text.")

    token_lens = count_tokens_batch(sentences, model=model)
    n = len(sentences)
    cumtok = _build_cumulative_tokens(token_lens)
    embeddings = get_embeddings(sentences)

    dp = [INF] * (n + 1)    # dp[j] = costo mínimo primeras j oraciones
    back = [-1] * (n + 1)  # punteros para reconstrucción por backtracking
    dp[0] = 0.0

    for j in range(1, n + 1):  # O(n^2)
        for i in range(j):
            span = cumtok[j] - cumtok[i]
            if span < lmin or span > lmax:
                continue
            cost = dp[i] + _compute_cost(
                token_count=span,
                embeddings=embeddings,
                start=i,
                end=j,
                coherence_lambda=coherence_lambda,
                fixed_cost=fixed_cost,
            )
            if cost < dp[j]:
                dp[j] = cost
                back[j] = i

    if dp[n] == INF:
        raise ValueError(
            f"No valid segmentation found with lmin={lmin}, lmax={lmax}. "
            f"Try relaxing the token constraints."
        )

    cuts = []
    cur = n
    while cur > 0:
        cuts.append(cur)
        cur = back[cur]
    cuts.reverse()

    segments = []
    prev = 0
    for idx, cut in enumerate(cuts):
        sents = sentences[prev:cut]
        segments.append(Segment(
            index=idx,
            sentences=sents,
            text=" ".join(sents),
            token_count=cumtok[cut] - cumtok[prev],
        ))
        prev = cut

    return SegmentationResult(
        segments=segments,
        total_cost=dp[n],
        num_segments=len(segments),
    )


# ── DP 2D — dp[i][k] ─────────────────────────────────────────────────────────

from dataclasses import dataclass


@dataclass
class DP2DResult:
    segments: list[Segment]
    total_cost: float
    num_segments: int
    cost_by_k: dict[int, float]


def segment_dp_2d(
    text: str,
    lmin: int = 50,
    lmax: int = 200,
    model: str = "gpt-4o",
    coherence_lambda: float = DEFAULT_LAMBDA,
    max_k: int = None,
    fixed_cost: float = DEFAULT_FIXED_COST,
) -> DP2DResult:
    sentences = split_sentences(text)
    if not sentences:
        raise ValueError("No sentences found in input text.")

    token_lens = count_tokens_batch(sentences, model=model)
    n = len(sentences)
    cumtok = _build_cumulative_tokens(token_lens)
    embeddings = get_embeddings(sentences)

    if max_k is None:
        max_k = n

    # dp[j][k] = costo mínimo de segmentar j oraciones en exactamente k segmentos
    dp = [[INF] * (max_k + 1) for _ in range(n + 1)]
    back = [[(-1, -1)] * (max_k + 1) for _ in range(n + 1)]
    dp[0][0] = 0.0

    for k in range(1, max_k + 1):
        for j in range(k, n + 1):
            for i in range(k - 1, j):
                span = cumtok[j] - cumtok[i]
                if span < lmin or span > lmax:
                    continue
                if dp[i][k - 1] == INF:
                    continue
                cost = dp[i][k - 1] + _compute_cost(
                    token_count=span,
                    embeddings=embeddings,
                    start=i,
                    end=j,
                    coherence_lambda=coherence_lambda,
                    fixed_cost=fixed_cost,
                )
                if cost < dp[j][k]:
                    dp[j][k] = cost
                    back[j][k] = (i, k - 1)

    # seleccionar k óptimo: argmin_k { dp[n][k] }
    cost_by_k = {}
    for k in range(1, max_k + 1):
        if dp[n][k] < INF:
            cost_by_k[k] = dp[n][k]

    if not cost_by_k:
        raise ValueError(
            f"No valid segmentation found with lmin={lmin}, lmax={lmax}. "
            f"Try relaxing the token constraints."
        )

    best_k = min(cost_by_k, key=cost_by_k.get)
    best_cost = cost_by_k[best_k]

    # backtracking desde dp[n][best_k]
    cuts = []
    cur = n
    cur_k = best_k
    while cur > 0:
        prev_i, prev_k = back[cur][cur_k]
        cuts.append(cur)
        cur = prev_i
        cur_k = prev_k
    cuts.reverse()

    segments = []
    prev = 0
    for idx, cut in enumerate(cuts):
        sents = sentences[prev:cut]
        segments.append(Segment(
            index=idx,
            sentences=sents,
            text=" ".join(sents),
            token_count=cumtok[cut] - cumtok[prev],
        ))
        prev = cut

    return DP2DResult(
        segments=segments,
        total_cost=best_cost,
        num_segments=best_k,
        cost_by_k=cost_by_k,
    )