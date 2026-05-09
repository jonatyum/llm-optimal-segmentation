# dp.py — algoritmo DP con función de costo compuesta
from src.segmentation.models import Segment, SegmentationResult
from src.segmentation.tokenizer import count_tokens_batch
from src.segmentation.splitter import split_sentences
from src.segmentation.embeddings import get_embeddings, cosine_similarity
import numpy as np

INF = float("inf")
DEFAULT_LAMBDA = 0.5        # peso de coherencia semántica
DEFAULT_FIXED_COST = 1000.0  # overhead fijo por llamada al LLM


def _build_similarity_arrays(embeddings: np.ndarray) -> tuple[list[float], list[float]]:
    """Retorna (sims, prefix_sims) donde sims[k] = cos(emb[k], emb[k+1]).

    prefix_sims[k] = sum(sims[0..k-1]) — permite calcular media de coherencia
    en O(1) usando _avg_coherence.
    """
    n = len(embeddings)
    sims: list[float] = []
    prefix_sims: list[float] = [0.0] * n
    for k in range(n - 1):
        s = cosine_similarity(embeddings[k], embeddings[k + 1])
        sims.append(s)
        prefix_sims[k + 1] = prefix_sims[k] + s
    return sims, prefix_sims


def _avg_coherence(prefix_sims: list[float], start: int, end: int) -> float:
    """Coherencia promedio del rango [start, end) en O(1).

    Cuenta pares (start, start+1), ..., (end-2, end-1) → n_pairs = end-start-1.
    prefix_sims[k] acumula similitudes 0..k-1, por lo que la suma del rango es
    prefix_sims[end-1] - prefix_sims[start].
    """
    n_pairs = end - start - 1
    if n_pairs <= 0:
        return 1.0
    return (prefix_sims[end - 1] - prefix_sims[start]) / n_pairs


def _compute_cost(
    token_count: int,
    prefix_sims: list[float],
    start: int,
    end: int,
    coherence_lambda: float,
    fixed_cost: float = DEFAULT_FIXED_COST,
) -> float:
    # Componente 1: costo cuadrático de autoatención
    t2 = float(token_count ** 2)

    # Componente 2: λ escala el t² — hace que el peso de coherencia sea proporcional al costo
    avg_coh = _avg_coherence(prefix_sims, start, end)
    coherence_factor = 1.0 + coherence_lambda * (1.0 - avg_coh)

    return t2 * coherence_factor + fixed_cost


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
    _, prefix_sims = _build_similarity_arrays(embeddings)

    dp = [INF] * (n + 1)    # dp[j] = costo mínimo primeras j oraciones
    back = [-1] * (n + 1)  # punteros para reconstrucción por backtracking
    dp[0] = 0.0

    for j in range(1, n + 1):  # O(n^2)
        for i in range(j):
            span = cumtok[j] - cumtok[i]
            if span > lmax:
                continue
            if span < lmin:
                break  # al avanzar i hacia j el span solo decrece
            cost = dp[i] + _compute_cost(
                token_count=span,
                prefix_sims=prefix_sims,
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
    _, prefix_sims = _build_similarity_arrays(embeddings)

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
                if span > lmax:
                    continue
                if span < lmin:
                    break  # span solo decrece al aumentar i
                if dp[i][k - 1] == INF:
                    continue
                cost = dp[i][k - 1] + _compute_cost(
                    token_count=span,
                    prefix_sims=prefix_sims,
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
    if back[n][best_k] == (-1, -1):
        raise ValueError(
            f"Estado de backtracking inválido en k={best_k}. No se encontró segmentación válida."
        )

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
