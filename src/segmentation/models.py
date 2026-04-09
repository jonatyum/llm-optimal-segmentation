# src/segmentation/models.py
from dataclasses import dataclass


@dataclass
class Segment:
    index: int
    sentences: list[str]
    text: str
    token_count: int


@dataclass
class SegmentationResult:
    segments: list[Segment]
    total_cost: float
    num_segments: int