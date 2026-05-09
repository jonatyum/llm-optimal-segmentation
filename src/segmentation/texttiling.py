# texttiling.py — TextTiling adaptado con embeddings semánticos (Hearst 1997, variante moderna)
import numpy as np

from src.segmentation.models import Segment, SegmentationResult
from src.segmentation.tokenizer import count_tokens_batch
from src.segmentation.splitter import split_sentences
from src.segmentation.embeddings import get_embeddings, cosine_similarity


def _block_centroid(embeddings: np.ndarray, start: int, end: int) -> np.ndarray:
    """Centroide medio de embeddings[start:end]. Retorna vector cero si el rango está vacío."""
    if end <= start:
        return np.zeros(embeddings.shape[1] if embeddings.ndim > 1 else 1)
    block = embeddings[start:end]
    return block.mean(axis=0)


def _smooth(values: list[float], passes: int) -> list[float]:
    """Media móvil de ventana 3 aplicada `passes` veces."""
    result = list(values)
    for _ in range(passes):
        smoothed = result[:]
        for i in range(1, len(result) - 1):
            smoothed[i] = (result[i - 1] + result[i] + result[i + 1]) / 3.0
        result = smoothed
    return result


def _is_local_minimum(values: list[float], k: int) -> bool:
    """Retorna True si values[k] es mínimo local estricto."""
    left_ok = k == 0 or values[k] <= values[k - 1]
    right_ok = k == len(values) - 1 or values[k] <= values[k + 1]
    strictly_less = (k > 0 and values[k] < values[k - 1]) or (
        k < len(values) - 1 and values[k] < values[k + 1]
    )
    return left_ok and right_ok and strictly_less


def segment_texttiling(
    text: str,
    lmin: int = 50,
    lmax: int = 200,
    model: str = "gpt-4o",
    window_size: int = 2,
    smoothing_passes: int = 1,
) -> SegmentationResult:
    """Segmentación TextTiling con comparación de bloques de embeddings semánticos.

    Posición k representa la frontera entre la oración k-1 y la oración k.
    block_sim[k] = coseno entre centroide de oraciones [k-window_size, k)
                   y centroide de oraciones [k, k+window_size).
    """
    sentences = split_sentences(text)
    if not sentences:
        raise ValueError("No sentences found in input text.")

    n = len(sentences)
    token_lens = count_tokens_batch(sentences, model=model)
    cumtok = [0] * (n + 1)
    for i, t in enumerate(token_lens):
        cumtok[i + 1] = cumtok[i] + t

    if n == 1:
        seg = Segment(index=0, sentences=sentences, text=sentences[0], token_count=token_lens[0])
        return SegmentationResult(
            segments=[seg],
            total_cost=float(token_lens[0] ** 2),
            num_segments=1,
        )

    embeddings = get_embeddings(sentences)

    # block_sim[k] para k in [1, n-1]: posición de corte entre oración k-1 y k
    block_sim: list[float] = []
    gap_positions: list[int] = []  # índice k del gap (corte antes de oración k)

    for k in range(1, n):
        left_start = max(0, k - window_size)
        right_end = min(n, k + window_size)
        left_centroid = _block_centroid(embeddings, left_start, k)
        right_centroid = _block_centroid(embeddings, k, right_end)
        sim = cosine_similarity(left_centroid, right_centroid)
        block_sim.append(sim)
        gap_positions.append(k)

    # Suavizado
    smoothed = _smooth(block_sim, smoothing_passes)

    # depth_score[i]: suma de descenso desde máximos locales vecinos hacia el mínimo
    m = len(smoothed)
    depth: list[float] = []
    for i in range(m):
        # máximo a la izquierda (incluyendo i)
        max_left = max(smoothed[: i + 1])
        # máximo a la derecha (incluyendo i)
        max_right = max(smoothed[i:])
        depth.append((max_left - smoothed[i]) + (max_right - smoothed[i]))

    depth = _smooth(depth, smoothing_passes)

    # candidatos: mínimos locales de depth (profundidad mayor = más candidatos)
    # Ordenar por depth descendente para seleccionar los más pronunciados primero
    candidates: list[int] = []
    for i in range(m):
        if _is_local_minimum(depth, i):
            candidates.append(gap_positions[i])

    # Si no hay mínimos locales, usar posición de máxima profundidad como único candidato
    if not candidates:
        best_i = int(np.argmax(depth))
        candidates = [gap_positions[best_i]]

    candidate_set = set(candidates)

    # Greedy: recorrer oraciones acumulando tokens; cortar en candidatos respetando lmin/lmax.
    # k is the index of the *next* sentence to be added; the segment so far is [seg_start, k).
    # We check whether adding sentence k would push us over lmax BEFORE adding it.
    cut_points: list[int] = []
    seg_start = 0

    for k in range(1, n):
        tokens_with_next = cumtok[k + 1] - cumtok[seg_start]
        tokens_so_far = cumtok[k] - cumtok[seg_start]

        if tokens_with_next > lmax and tokens_so_far >= lmin:
            # Adding sentence k would violate lmax; cut before sentence k
            cut_points.append(k)
            seg_start = k
            continue

        if tokens_so_far >= lmax:
            # Force cut even if segment would be under lmin (unavoidable)
            cut_points.append(k)
            seg_start = k
            continue

        if tokens_so_far >= lmin and k in candidate_set:
            cut_points.append(k)
            seg_start = k

    # Construir segmentos
    boundaries = cut_points + [n]
    segments: list[Segment] = []
    prev = 0
    for idx, cut in enumerate(boundaries):
        sents = sentences[prev:cut]
        if not sents:
            continue
        tok = cumtok[cut] - cumtok[prev]
        segments.append(Segment(
            index=len(segments),
            sentences=sents,
            text=" ".join(sents),
            token_count=tok,
        ))
        prev = cut

    if not segments:
        seg = Segment(index=0, sentences=sentences, text=" ".join(sentences), token_count=cumtok[n])
        segments = [seg]

    total_cost = float(sum(s.token_count ** 2 for s in segments))
    return SegmentationResult(segments=segments, total_cost=total_cost, num_segments=len(segments))
