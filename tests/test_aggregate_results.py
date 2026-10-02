"""Tests for build_aggregate and its helpers in reasonance.utils."""

import json

import pytest
from reasonance.utils import (
    _RESULT_COLUMNS,
    _is_instruction_tuned,
    _model_key_to_name,
    _parse_result_file,
    build_aggregate,
)

# --- unit tests for small helpers ---


def test_model_key_to_name():
    assert _model_key_to_name("Qwen--Qwen3-4B") == "Qwen/Qwen3-4B"
    assert _model_key_to_name("DeepSeek-R1") == "DeepSeek-R1"


def test_is_instruction_tuned_known_api_models():
    assert _is_instruction_tuned("o3") is True
    assert _is_instruction_tuned("DeepSeek-R1") is True
    assert _is_instruction_tuned("claude-opus-4-5") is True


def test_is_instruction_tuned_by_name_pattern():
    assert _is_instruction_tuned("meta-llama--Meta-Llama-3-8B-Instruct") is True
    assert _is_instruction_tuned("Qwen--Qwen2.5-7B-Instruct") is True
    assert _is_instruction_tuned("Qwen--Qwen3-4B") is False


# --- integration test using a minimal fake results tree ---

_MINIMAL_RESULT = {
    "accuracy": 0.75,
    "balanced_accuracy": 0.76,
    "threshold": 0.5,
    "threshold_fitted_on": 0,
    "threshold_obj": "balanced_accuracy",
    "predictions_path": "/some/root/results/reasoning/0-bullet-is/model-DeepSeek-R1/DeepSeek-R1_task-ACSIncome/DeepSeek-R1_bench-123/ACSIncome.test_predictions.csv",  # noqa: E501
    "config": {
        "few_shot": 0,
        "correct_order_bias": True,
        "reasoning": 1,
        "prompt_variation": {
            "format": "bullet",
            "connector": "is",
            "granularity": "original",
            "order": None,
        },
    },
}


@pytest.fixture()
def fake_results_dir(tmp_path):
    """Create a minimal results tree with one result JSON under a results/ root."""
    bench_dir = tmp_path / "results" / "model-DeepSeek-R1_task-ACSIncome" / "DeepSeek-R1_bench-123"
    bench_dir.mkdir(parents=True)
    (bench_dir / "results.bench-123.json").write_text(json.dumps(_MINIMAL_RESULT))
    return tmp_path / "results"


def test_parse_result_file(fake_results_dir):
    json_file = (
        fake_results_dir / "model-DeepSeek-R1_task-ACSIncome" / "DeepSeek-R1_bench-123" / "results.bench-123.json"
    )
    row = _parse_result_file(json_file, task="ACSIncome")
    assert row is not None
    assert row["model"] == "DeepSeek-R1"
    assert row["task"] == "ACSIncome"
    assert row["is_reasoning_model"] == 1
    assert row["reasoning_effort"] == 1
    assert row["accuracy"] == pytest.approx(0.75)
    # paths must be relative
    assert row["eval_results_path"].startswith("results/")
    assert row["predictions_path"].startswith("results/")
    assert row["responses_path"].endswith(".test_responses.csv")


def test_build_aggregate_columns(fake_results_dir):
    df = build_aggregate(fake_results_dir, tasks="ACSIncome")
    assert list(df.columns) == _RESULT_COLUMNS


def test_build_aggregate_shape(fake_results_dir):
    df = build_aggregate(fake_results_dir, tasks="ACSIncome")
    assert len(df) == 1


def test_build_aggregate_save(fake_results_dir, tmp_path):
    out = tmp_path / "out.csv"
    build_aggregate(fake_results_dir, tasks="ACSIncome", save_path=out)
    assert out.exists()
    import pandas as pd

    df = pd.read_csv(out)
    assert len(df) == 1
