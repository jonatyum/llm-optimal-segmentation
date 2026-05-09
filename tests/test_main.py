import sys
import pytest
from unittest.mock import patch
from main import main

CLI_TEXT = (
    "Large language models have billions of parameters and require significant resources. "
    "Dynamic programming finds globally optimal solutions to segmentation problems. "
    "Text segmentation divides documents into smaller chunks for LLM processing. "
    "Semantic coherence measures how well consecutive sentences relate to each other. "
    "Transformers use self-attention mechanisms to process all tokens in parallel. "
    "Retrieval-augmented generation combines language models with external knowledge. "
    "Evaluation of language models remains an open and challenging research problem."
)

_BASE_ARGS = ["main.py", "--text", CLI_TEXT, "--lmin", "10", "--lmax", "60"]


def _run(*args):
    with patch("sys.argv", [*_BASE_ARGS, *args]):
        main()


def test_segment_dp(capsys):
    _run("segment", "--method", "dp")
    assert "DP" in capsys.readouterr().out


def test_segment_baseline(capsys):
    _run("segment", "--method", "baseline")
    assert "BASELINE" in capsys.readouterr().out


def test_segment_sliding_window(capsys):
    _run("segment", "--method", "sliding_window")
    assert "SLIDING_WINDOW" in capsys.readouterr().out


def test_segment_overlap(capsys):
    _run("segment", "--method", "overlap")
    assert "OVERLAP" in capsys.readouterr().out


def test_segment_dp2d(capsys):
    _run("segment", "--method", "dp2d")
    assert "k óptimo" in capsys.readouterr().out


def test_compare(capsys):
    _run("compare")
    assert "Reducción de costo" in capsys.readouterr().out


def test_report_no_llm(capsys):
    _run("report")
    out = capsys.readouterr().out
    assert "Tokens totales" in out
    assert "k* analítico" in out


def test_calibrate(capsys):
    _run("calibrate")
    out = capsys.readouterr().out
    assert "λ sugerido" in out
    assert "k* analítico" in out


def test_no_command_prints_help(capsys):
    with patch("sys.argv", ["main.py", "--text", CLI_TEXT]):
        main()
    out = capsys.readouterr().out
    assert len(out) > 0
