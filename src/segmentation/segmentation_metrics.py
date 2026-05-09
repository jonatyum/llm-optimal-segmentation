# segmentation_metrics.py — métricas estándar de evaluación de segmentación de texto
#
# Pk  : Beeferman et al. (1999). Penaliza errores de boundary a distancia k.
# WD  : Pevzner & Hearst (2002). WindowDiff — cuenta boundaries dentro de ventana.
#
# Convención: boundaries es una lista de 0/1 de longitud (N_oraciones - 1),
# donde boundaries[i] = 1 significa que hay un corte DESPUÉS de la oración i.

from __future__ import annotations

import math


def boundaries_from_result(result) -> list[int]:
    """Convierte SegmentationResult o OverlapSegmentationResult a lista de boundaries.

    Retorna lista de longitud (total_oraciones - 1).
    boundaries[i] = 1 ↔ hay un corte entre la oración i y la i+1.
    """
    boundaries: list[int] = []
    segments = result.segments
    for seg_idx, seg in enumerate(segments[:-1]):  # omitir el último: no hay corte después
        n_sents = len(seg.sentences)
        # las n_sents - 1 oraciones internas del segmento no tienen corte
        boundaries.extend([0] * (n_sents - 1))
        # la última oración del segmento sí tiene corte hacia el siguiente
        boundaries.append(1)
    # las oraciones del último segmento no tienen corte posterior
    if segments:
        boundaries.extend([0] * (len(segments[-1].sentences) - 1))
    return boundaries


def _cumulative_cuts(boundaries: list[int]) -> list[int]:
    """Suma acumulada de boundaries — permite contar cortes en O(1) por ventana."""
    cum = [0] * (len(boundaries) + 1)
    for i, b in enumerate(boundaries):
        cum[i + 1] = cum[i] + b
    return cum


def _default_k(reference: list[int]) -> int:
    """k por defecto = mitad del tamaño promedio de segmento (en oraciones)."""
    total_sentences = len(reference) + 1
    n_boundaries = sum(reference)
    n_segments = n_boundaries + 1
    avg_seg_size = total_sentences / n_segments
    return max(1, round(avg_seg_size / 2))


def _same_segment(cum: list[int], i: int, j: int) -> bool:
    """True si las oraciones i y j están en el mismo segmento (no hay corte entre ellas)."""
    return cum[j] - cum[i] == 0


def pk(reference: list[int], hypothesis: list[int], k: int | None = None) -> float:
    """Métrica Pk (Beeferman et al. 1999). Menor es mejor; 0 es perfecto.

    Desliza una ventana de tamaño k sobre las oraciones. Para cada par (i, i+k)
    compara si están en el mismo segmento en la referencia vs en la hipótesis.
    Penaliza los desacuerdos.

    Args:
        reference:  lista de boundaries de la segmentación de referencia
        hypothesis: lista de boundaries de la segmentación propuesta
        k:          tamaño de ventana; por defecto = mitad del segmento promedio

    Returns:
        Proporción de ventanas en las que referencia e hipótesis discrepan.
    """
    if len(reference) != len(hypothesis):
        raise ValueError(
            f"reference y hypothesis deben tener la misma longitud "
            f"({len(reference)} != {len(hypothesis)})"
        )
    if not reference:
        return 0.0

    if k is None:
        k = _default_k(reference)

    n_sentences = len(reference) + 1
    if k >= n_sentences:
        k = max(1, n_sentences - 1)

    cum_ref = _cumulative_cuts(reference)
    cum_hyp = _cumulative_cuts(hypothesis)

    errors = 0
    n_windows = n_sentences - k
    if n_windows <= 0:
        return 0.0

    for i in range(n_windows):
        j = i + k
        ref_same = _same_segment(cum_ref, i, j)
        hyp_same = _same_segment(cum_hyp, i, j)
        if ref_same != hyp_same:
            errors += 1

    return errors / n_windows


def window_diff(reference: list[int], hypothesis: list[int], k: int | None = None) -> float:
    """WindowDiff (Pevzner & Hearst 2002). Menor es mejor; 0 es perfecto.

    Desliza una ventana de tamaño k. Para cada posición i, compara el número de
    boundaries dentro de la ventana [i, i+k) en referencia vs hipótesis.
    Penaliza cualquier diferencia (incluso si ambas contienen boundaries pero en
    posiciones distintas).

    Args:
        reference:  lista de boundaries de la segmentación de referencia
        hypothesis: lista de boundaries de la segmentación propuesta
        k:          tamaño de ventana; por defecto = mitad del segmento promedio

    Returns:
        Proporción de ventanas en las que el conteo de boundaries difiere.
    """
    if len(reference) != len(hypothesis):
        raise ValueError(
            f"reference y hypothesis deben tener la misma longitud "
            f"({len(reference)} != {len(hypothesis)})"
        )
    if not reference:
        return 0.0

    if k is None:
        k = _default_k(reference)

    n_sentences = len(reference) + 1
    if k >= n_sentences:
        k = max(1, n_sentences - 1)

    cum_ref = _cumulative_cuts(reference)
    cum_hyp = _cumulative_cuts(hypothesis)

    errors = 0
    n_windows = n_sentences - k
    if n_windows <= 0:
        return 0.0

    for i in range(n_windows):
        j = i + k
        # número de boundaries dentro de la ventana (entre posiciones i y j)
        ref_count = cum_ref[j] - cum_ref[i]
        hyp_count = cum_hyp[j] - cum_hyp[i]
        if ref_count != hyp_count:
            errors += 1

    return errors / n_windows
