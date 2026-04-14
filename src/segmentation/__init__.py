from src.segmentation.dp import segment_dp, segment_dp_2d, DP2DResult
from src.segmentation.baseline import segment_baseline
from src.segmentation.overlap import segment_dp_overlap
from src.segmentation.sliding_window import segment_sliding_window
from src.segmentation.metrics import compute_metrics, compare
from src.segmentation.embeddings import segment_coherence, get_embeddings
from src.segmentation.tokenizer import count_tokens, count_tokens_batch
from src.segmentation.splitter import split_sentences
from src.segmentation.evaluator import evaluate_segmentation, compare_evaluations
from src.segmentation.report import generate_full_report, FullReport
from src.segmentation.models import Segment, SegmentationResult
from src.segmentation.calibration import (
    suggest_lambda,
    estimate_optimal_k,
    calibrate_fixed_cost,
    validate_calibration,
)

__all__ = [
    "segment_dp",
    "segment_dp_2d",
    "DP2DResult",
    "segment_baseline",
    "segment_dp_overlap",
    "segment_sliding_window",
    "compute_metrics",
    "compare",
    "segment_coherence",
    "get_embeddings",
    "count_tokens",
    "count_tokens_batch",
    "split_sentences",
    "evaluate_segmentation",
    "compare_evaluations",
    "generate_full_report",
    "FullReport",
    "Segment",
    "SegmentationResult",
    "suggest_lambda",
    "estimate_optimal_k",
    "calibrate_fixed_cost",
    "validate_calibration",
]