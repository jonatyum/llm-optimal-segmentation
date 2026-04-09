from src.segmentation.models import Segment, SegmentationResult
from src.segmentation.tokenizer import count_tokens_batch
from src.segmentation.splitter import split_sentences

INF = float("inf")


def _compute_cost(token_count: int) -> float:
    return float(token_count ** 2)


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
) -> SegmentationResult:
    sentences = split_sentences(text)
    if not sentences:
        raise ValueError("No sentences found in input text.")

    token_lens = count_tokens_batch(sentences, model=model)
    n = len(sentences)
    cumtok = _build_cumulative_tokens(token_lens)

    dp = [INF] * (n + 1)
    back = [-1] * (n + 1)
    dp[0] = 0.0

    for j in range(1, n + 1):
        for i in range(j):
            span = cumtok[j] - cumtok[i]
            if span < lmin or span > lmax:
                continue
            cost = dp[i] + _compute_cost(span)
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