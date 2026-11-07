"""Monitoring tests: device info, VRAM reporting, and the benchmark script."""

import json
import subprocess
import sys

import pytest
import torch

from model.atlas_llm import AtlasLLM
from model.config import ModelConfig
from monitoring.system import device_info, vram_usage
from scripts.benchmark import bench_step

TINY_MODEL = {
    "vocab_size": 32,
    "context_length": 8,
    "d_model": 16,
    "n_layers": 1,
    "n_heads": 2,
    "d_ff": 32,
}


def tiny_model() -> AtlasLLM:
    return AtlasLLM(ModelConfig(**TINY_MODEL))


# ------------------------------------------------------------------- device


def test_device_info_on_cpu():
    info = device_info(torch.device("cpu"))
    assert info.type == "cpu"
    assert info.name
    assert info.total_memory_mb == 0.0
    assert info.cuda_version is None
    assert info.capability is None
    assert info.torch_version == torch.__version__


def test_device_info_to_dict_is_json_ready():
    payload = device_info(torch.device("cpu")).to_dict()
    assert set(payload) == {"type", "name", "total_memory_mb", "cuda_version", "capability", "torch_version"}
    json.dumps(payload)


def test_vram_usage_is_none_on_cpu():
    assert vram_usage(torch.device("cpu")) is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_vram_usage_reports_a_consistent_snapshot_on_cuda():
    usage = vram_usage(torch.device("cuda"))
    assert usage is not None
    assert usage.total_mb > 0
    # allocated <= reserved always; peak >= allocated since the last reset.
    # reserved/peak may exceed total_mb on Windows WDDM (VRAM is overcommitted
    # through host memory instead of failing), so that is not asserted.
    assert 0.0 <= usage.allocated_mb <= usage.reserved_mb
    assert usage.peak_mb >= usage.allocated_mb
    assert set(usage.to_dict()) == {"allocated_mb", "reserved_mb", "peak_mb", "total_mb"}


# ----------------------------------------------------------------- benchmark


def test_bench_step_counts_tokens_and_runs_the_real_step():
    row = bench_step(
        tiny_model(),
        batch_size=2,
        seq_len=8,
        vocab_size=TINY_MODEL["vocab_size"],
        steps=2,
        warmup=0,
        device=torch.device("cpu"),
        learning_rate=1e-3,
        weight_decay=0.0,
    )
    assert row["ok"] is True and row["oom"] is False
    assert row["spills"] is False
    assert row["batch_size"] == 2 and row["seq_len"] == 8
    assert row["tokens_per_step"] == 16
    assert row["ms_per_step"] > 0
    assert row["tokens_per_sec"] > 0
    # Cross-entropy on a fresh model sits near ln(V); a broken step would not.
    assert 1.0 < row["final_loss"] < 10.0
    # CPU has no device memory, so the same zero convention as the trainer.
    assert row["allocated_mb"] == 0.0 and row["reserved_mb"] == 0.0


def test_bench_step_records_oom_instead_of_crashing(monkeypatch):
    import scripts.benchmark as benchmark

    def out_of_memory(*args, **kwargs):
        raise torch.OutOfMemoryError("simulated")

    monkeypatch.setattr(benchmark, "lm_cross_entropy", out_of_memory)
    row = bench_step(
        tiny_model(),
        batch_size=2,
        seq_len=8,
        vocab_size=TINY_MODEL["vocab_size"],
        steps=1,
        warmup=0,
        device=torch.device("cpu"),
        learning_rate=1e-3,
        weight_decay=0.0,
    )
    assert row["ok"] is False and row["oom"] is True
    assert row["spills"] is None
    assert row["tokens_per_sec"] is None and row["peak_allocated_mb"] is None


def test_cli_writes_a_json_report(tmp_path):
    config_path = tmp_path / "tiny.yaml"
    config_path.write_text(
        """
model:
  vocab_size: 32
  context_length: 8
  d_model: 16
  n_layers: 1
  n_heads: 2
  d_ff: 32
training:
  learning_rate: 0.001
  weight_decay: 0.0
device: cpu
dtype: float32
""",
        encoding="utf-8",
    )
    out = tmp_path / "bench.json"
    result = subprocess.run(
        [
            sys.executable, "-m", "scripts.benchmark",
            "--config", str(config_path),
            "--batch-sizes", "1", "2",
            "--steps", "1", "--warmup", "0",
            "--out", str(out),
        ],
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, result.stderr

    artifact = json.loads(out.read_text(encoding="utf-8"))
    assert [row["batch_size"] for row in artifact["results"]] == [1, 2]
    assert all(row["ok"] for row in artifact["results"])
    assert artifact["device"]["type"] == "cpu"
    assert artifact["parameters"] > 0
    assert artifact["seq_len"] == 8
