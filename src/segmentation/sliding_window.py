from dataclasses import dataclass
from src.segmentation.models import SegmentationResult
from src.segmentation.tokenizer import count_tokens_batch
from src.segmentation.splitter import split_sentences


@dataclass
class SlidingWindowSegment:
    index: int
    sentences: list[str]
    text: str
    token_count: int
    overlap_tokens: int


@dataclass
class SlidingWindowResult:
    segments: list[SlidingWindowSegment]
    total_cost: float
    num_segments: int
    total_overlap_tokens: int


def segment_sliding_window(
    text: str,
    lmax: int = 200,
    overlap: int = 50,
    model: str = "gpt-4o",
) -> SlidingWindowResult:
    if overlap >= lmax:
        raise ValueError(
            f"overlap={overlap} must be smaller than lmax={lmax}."
        )

    sentences = split_sentences(text)
    if not sentences:
        raise ValueError("No sentences found in input text.")

    token_lens = count_tokens_batch(sentences, model=model)
    n = len(sentences)

    segments = []
    seg_index = 0
    i = 0

    while i < n:
        current_sentences = []
        current_tokens = 0
        j = i

        # acumular oraciones hasta lmax
        while j < n and current_tokens + token_lens[j] <= lmax:
            current_sentences.append(sentences[j])
            current_tokens += token_lens[j]
            j += 1

        if not current_sentences:
            current_sentences.append(sentences[j])
            current_tokens = token_lens[j]
            j += 1

        # calcular overlap — retroceder desde i hasta cubrir overlap tokens
        overlap_sentences = []
        overlap_tokens = 0
        if seg_index > 0:
            k = i - 1
            while k >= 0 and overlap_tokens + token_lens[k] <= overlap:
                overlap_sentences.insert(0, sentences[k])
                overlap_tokens += token_lens[k]
                k -= 1

        segments.append(SlidingWindowSegment(
            index=seg_index,
            sentences=current_sentences,
            text=" ".join(current_sentences),
            token_count=current_tokens,
            overlap_tokens=overlap_tokens,
        ))

        seg_index += 1

        # avanzar saltando las oraciones del overlap
        overlap_skip = 0
        overlap_count = 0
        k = i
        while k < j and overlap_count + token_lens[k] <= overlap:
            overlap_count += token_lens[k]
            overlap_skip += 1
            k += 1

        step = max(1, len(current_sentences) - overlap_skip)
        i += step

    total_cost = sum(s.token_count ** 2 for s in segments)
    total_overlap_tokens = sum(s.overlap_tokens for s in segments)

    return SlidingWindowResult(
        segments=segments,
        total_cost=total_cost,
        num_segments=len(segments),
        total_overlap_tokens=total_overlap_tokens,
    )