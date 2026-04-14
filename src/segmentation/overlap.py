from dataclasses import dataclass
from src.segmentation.models import Segment, SegmentationResult
from src.segmentation.tokenizer import count_tokens_batch
from src.segmentation.splitter import split_sentences
from src.segmentation.embeddings import get_embeddings, cosine_similarity
import numpy as np

INF = float("inf")
DEFAULT_LAMBDA = 0.5
DEFAULT_MU = 0.3
DEFAULT_MAX_OVERLAP = 3
DEFAULT_FIXED_COST = 100.0


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
    embeddings: np.ndarray,
    start: int,
    end: int,
    coherence_lambda: float,
    overlap_mu: float,
    fixed_cost: float = DEFAULT_FIXED_COST,
) -> float:
    computational_cost = float(token_count ** 2)

    if end - start < 2:
        coherence_penalty = 0.0
    else:
        scores = []
        for i in range(start, end - 1):
            score = cosine_similarity(embeddings[i], embeddings[i + 1])
            scores.append(score)
        avg_coherence = float(np.mean(scores))
        coherence_penalty = 1.0 - avg_coherence

    overlap_cost = float(overlap_tokens ** 2)

    return (
        computational_cost
        + coherence_lambda * coherence_penalty
        + overlap_mu * overlap_cost
        + fixed_cost
    )


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

    # dp[j][o] = costo minimo para segmentar hasta oracion j con overlap o
    dp = [[INF] * (max_overlap + 1) for _ in range(n + 1)]
    back = [[(-1, 0)] * (max_overlap + 1) for _ in range(n + 1)]
    dp[0][0] = 0.0

    for j in range(1, n + 1):
        for i in range(j):
            span = cumtok[j] - cumtok[i]
            if span < lmin or span > lmax:
                continue
            for o in range(max_overlap + 1):
                overlap_start = max(0, i - o)
                overlap_tokens = cumtok[i] - cumtok[overlap_start]
                prev_o = 0
                if dp[i][prev_o] == INF:
                    continue
                cost = dp[i][prev_o] + _compute_cost(
                    token_count=span,
                    overlap_tokens=overlap_tokens,
                    embeddings=embeddings,
                    start=overlap_start,
                    end=j,
                    coherence_lambda=coherence_lambda,
                    overlap_mu=overlap_mu,
                    fixed_cost=fixed_cost,
                )
                if cost < dp[j][o]:
                    dp[j][o] = cost
                    back[j][o] = (i, o)

    # encontrar el overlap optimo para dp[n]
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

    # backtracking
    cuts = []
    cur = n
    cur_o = best_o
    while cur > 0:
        prev_i, prev_o = back[cur][cur_o]
        cuts.append((cur, cur_o))
        cur = prev_i
        cur_o = prev_o
    cuts.reverse()

    # construir segmentos con overlap
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