# overlap.py — DP con overlap como variable de optimización
from dataclasses import dataclass
from src.segmentation.models import Segment, SegmentationResult
from src.segmentation.tokenizer import count_tokens_batch
from src.segmentation.splitter import split_sentences
from src.segmentation.embeddings import get_embeddings, cosine_similarity
from src.segmentation.dp import _build_similarity_arrays, _avg_coherence
import numpy as np

INF = float("inf")
DEFAULT_LAMBDA = 0.5
DEFAULT_MU = 0.03
DEFAULT_MAX_OVERLAP = 3
DEFAULT_FIXED_COST = 1000.0


@dataclass
class OverlapSegment:
    index: int
    sentences: list[str]
    text: str
    token_count: int
    overlap_sentences: list[str]
    overlap_tokens: int


@dataclass
class OverlapSegmentationResult:
    segments: list[OverlapSegment]
    total_cost: float
    num_segments: int
    total_overlap_tokens: int


def _compute_cost(
    token_count: int,
    overlap_tokens: int,
    sims: list[float],
    prefix_sims: list[float],
    start: int,
    end: int,
    boundary: int,
    coherence_lambda: float,
    overlap_mu: float,
    fixed_cost: float = DEFAULT_FIXED_COST,
) -> float:
    t2 = float(token_count ** 2)

    avg_coh = _avg_coherence(prefix_sims, start, end)
    coherence_factor = 1.0 + coherence_lambda * (1.0 - avg_coh)

    # Costo neto de overlap: penalidad lineal menos bonus por coherencia en el borde.
    # boundary_sim usa sims[boundary-1] = cos(emb[boundary-1], emb[boundary])
    overlap_cost = 0.0
    if overlap_tokens > 0 and boundary > 0 and boundary - 1 < len(sims):
        boundary_sim = sims[boundary - 1]
        overlap_cost = overlap_mu * overlap_tokens - coherence_lambda * boundary_sim

    return t2 * coherence_factor + overlap_cost + fixed_cost


def _build_cumulative_tokens(token_lens: list[int]) -> list[int]:
    cumulative = [0]
    for t in token_lens:
        cumulative.append(cumulative[-1] + t)
    return cumulative


def segment_dp_overlap(
    text: str,
    lmin: int = 50,
    lmax: int = 200,
    model: str = "gpt-4o",
    coherence_lambda: float = DEFAULT_LAMBDA,
    overlap_mu: float = DEFAULT_MU,
    max_overlap: int = DEFAULT_MAX_OVERLAP,
    fixed_cost: float = DEFAULT_FIXED_COST,
) -> OverlapSegmentationResult:
    sentences = split_sentences(text)
    if not sentences:
        raise ValueError("No sentences found in input text.")

    token_lens = count_tokens_batch(sentences, model=model)
    n = len(sentences)
    cumtok = _build_cumulative_tokens(token_lens)
    embeddings = get_embeddings(sentences)
    sims, prefix_sims = _build_similarity_arrays(embeddings)

    # dp[j][o] = costo mínimo llegando a oración j con o oraciones de overlap
    dp = [[INF] * (max_overlap + 1) for _ in range(n + 1)]
    back = [[(-1, -1)] * (max_overlap + 1) for _ in range(n + 1)]
    dp[0][0] = 0.0

    for j in range(1, n + 1):
        for i in range(j):
            span = cumtok[j] - cumtok[i]
            if span > lmax:
                continue
            if span < lmin:
                break  # span decrece al crecer i
            for o in range(max_overlap + 1):
                overlap_start = max(0, i - o)
                overlap_tokens = cumtok[i] - cumtok[overlap_start]
                # FIX 1: iterar sobre todos los prev_o válidos en dp[i]
                for prev_o in range(max_overlap + 1):
                    if dp[i][prev_o] == INF:
                        continue
                    cost = dp[i][prev_o] + _compute_cost(
                        token_count=span,
                        overlap_tokens=overlap_tokens,
                        sims=sims,
                        prefix_sims=prefix_sims,
                        start=overlap_start,
                        end=j,
                        boundary=i,
                        coherence_lambda=coherence_lambda,
                        overlap_mu=overlap_mu,
                        fixed_cost=fixed_cost,
                    )
                    if cost < dp[j][o]:
                        dp[j][o] = cost
                        back[j][o] = (i, prev_o)  # guardar prev_o real

    # seleccionar overlap óptimo para dp[n]
    best_cost = INF
    best_o = 0
    for o in range(max_overlap + 1):
        if dp[n][o] < best_cost:
            best_cost = dp[n][o]
            best_o = o

    if best_cost == INF:
        raise ValueError(
            f"No valid segmentation found with lmin={lmin}, lmax={lmax}. "
            f"Try relaxing the token constraints."
        )

    # backtracking — reconstruir segmentos con overlap
    cuts = []
    cur = n
    cur_o = best_o
    while cur > 0:
        prev_i, prev_o = back[cur][cur_o]
        cuts.append((cur, cur_o))
        cur = prev_i
        cur_o = prev_o
    cuts.reverse()

    # construir objetos OverlapSegment con metadatos de overlap
    segments = []
    prev = 0
    total_overlap_tokens = 0

    for idx, (cut, o) in enumerate(cuts):
        overlap_start = max(0, prev - o)
        overlap_sents = sentences[overlap_start:prev]
        overlap_toks = cumtok[prev] - cumtok[overlap_start]
        sents = sentences[prev:cut]

        segments.append(OverlapSegment(
            index=idx,
            sentences=sents,
            text=" ".join(sents),
            token_count=cumtok[cut] - cumtok[prev],
            overlap_sentences=overlap_sents,
            overlap_tokens=overlap_toks,
        ))
        total_overlap_tokens += overlap_toks
        prev = cut

    return OverlapSegmentationResult(
        segments=segments,
        total_cost=best_cost,
        num_segments=len(segments),
        total_overlap_tokens=total_overlap_tokens,
    )
