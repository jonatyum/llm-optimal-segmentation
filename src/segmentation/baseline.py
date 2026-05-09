# baseline.py — chunking fijo greedy O(n)
from src.segmentation.models import Segment, SegmentationResult
from src.segmentation.tokenizer import count_tokens_batch
from src.segmentation.splitter import split_sentences


def segment_baseline(
    text: str,
    lmax: int = 200,
    model: str = "gpt-4o",
) -> SegmentationResult:
    sentences = split_sentences(text)
    if not sentences:
        raise ValueError("No sentences found in input text.")

    token_lens = count_tokens_batch(sentences, model=model)

    segments = []
    current_sentences = []
    current_tokens = 0
    seg_index = 0

    for sentence, tok_len in zip(sentences, token_lens):
        # cortar si agregar esta oración supera lmax
        if current_tokens + tok_len > lmax and current_sentences:
            segments.append(Segment(
                index=seg_index,
                sentences=current_sentences,
                text=" ".join(current_sentences),
                token_count=current_tokens,
            ))
            seg_index += 1
            current_sentences = []
            current_tokens = 0

        current_sentences.append(sentence)
        current_tokens += tok_len

    if current_sentences:
        segments.append(Segment(
            index=seg_index,
            sentences=current_sentences,
            text=" ".join(current_sentences),
            token_count=current_tokens,
        ))

    # costo total = suma de tokens² de cada segmento
    total_cost = sum(s.token_count ** 2 for s in segments)

    return SegmentationResult(
        segments=segments,
        total_cost=total_cost,
        num_segments=len(segments),
    )