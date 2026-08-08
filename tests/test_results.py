"""Published-result gates for the real Liquid LFM benchmark."""

import json
from pathlib import Path

from online_sdft.config import ICL_K, METHODS, MODEL_ID, RAG_K


ROOT = Path(__file__).resolve().parents[1]


def test_published_run_uses_liquid_lfm_and_online_protocol():
    payload = json.loads(
        (ROOT / "outputs" / "bandit" / "summary.json").read_text()
    )
    config = payload["config"]
    assert config["student_model"] == MODEL_ID
    assert config["evaluation"] == "prequential one-stream; predict then learn"
    assert config["learning_signal"] == (
        "teacher rollouts only; oracle scoring only"
    )
    assert config["online_batch_size"] <= 4
    assert config["icl_examples"] == ICL_K
    assert config["rag_examples"] == RAG_K == ICL_K


def test_online_sdft_beats_every_published_baseline_on_both_metrics():
    summary = json.loads(
        (ROOT / "outputs" / "bandit" / "summary.json").read_text()
    )["summary"]
    baselines = METHODS[:-1]
    sdft = summary["Online-SDFT"]
    assert sdft["online_accuracy"]["mean"] > max(
        summary[method]["online_accuracy"]["mean"]
        for method in baselines
    )
    assert sdft["cum_regret"]["mean"] < min(
        summary[method]["cum_regret"]["mean"]
        for method in baselines
    )
