import copy
import json
from pathlib import Path
import subprocess
import sys

import pytest

from research import run


@pytest.fixture
def protocol():
    p = run.read_json(run.ROOT / "research/protocol.json")
    p.update(cases=["en-short"], seeds=[42], timing_requests=3, settle_seconds=0)
    return p


def measurements(baseline=100, candidate=90):
    return {"en-short/seed42": {
        "case": "en-short", "seed": 42, "exact": True,
        "absolute_failures": {"baseline": {}, "candidate": {}},
        "baseline": {"median_ms": baseline, "p90_ms": baseline, "rtf": baseline / 1000},
        "candidate": {"median_ms": candidate, "p90_ms": candidate, "rtf": candidate / 1000},
    }}


def test_requires_repeatable_gain_and_preserves_absolute_failures(protocol):
    quality = measurements()
    quality["en-short/seed42"]["absolute_failures"] = {
        "baseline": {"final_latent": .0792}, "candidate": {"final_latent": .0792}}
    result = run.decide(quality, [measurements() for _ in range(3)], protocol)
    assert result["status"] == "keep"
    assert result["gain"] == pytest.approx(.10)
    assert result["absolute_qualification"] == "partial"
    assert result["absolute_failures"]


@pytest.mark.parametrize("rounds,status", [
    ([measurements(100, 100)] * 3, "inconclusive"),
    ([measurements(100, 98)] * 3, "inconclusive"),
    ([measurements(100, 90), measurements(120, 108), measurements(110, 99)], "inconclusive"),
    ([measurements(100, 90), measurements(100, 110), measurements(100, 90)], "inconclusive"),
    ([measurements(100, 110)] * 3, "discard"),
])
def test_control_noise_order_effect_and_regression(protocol, rounds, status):
    assert run.decide(measurements(), rounds, protocol)["status"] == status


def test_tail_regression_blocks_a_fast_median(protocol):
    rounds = [measurements() for _ in range(3)]
    for block in rounds:
        block["en-short/seed42"]["candidate"]["p90_ms"] = 120
    assert run.decide(measurements(), rounds, protocol)["status"] == "discard"


@pytest.mark.parametrize("problem", ["audio", "new_failure", "worse_failure"])
def test_accuracy_cannot_be_traded_for_speed(protocol, problem):
    quality = measurements()
    row = quality["en-short/seed42"]
    if problem == "audio":
        row["exact"] = False
    else:
        row["absolute_failures"]["candidate"] = {"spectrum": .2}
        if problem == "worse_failure":
            row["absolute_failures"]["baseline"] = {"spectrum": .15}
    assert run.decide(quality, [measurements()] * 3, protocol)["status"] == "discard"


def test_incomplete_rounds_and_missing_seeds_fail_closed(protocol):
    with pytest.raises(ValueError, match="Incomplete timing"):
        run.decide(measurements(), [measurements()], protocol)
    protocol["seeds"].append(43)
    with pytest.raises(ValueError, match="Incomplete quality"):
        run.decide(measurements(), [measurements()] * 3, protocol)


def write_batch(directory, p):
    case = next(c for c in run.read_json(run.ROOT / "scripts/optimization_cases.json")["qualification"] if c["id"] == "en-short")
    report = {**case, "seed": 42, "steps": p["steps"], "speed": p["speed"], "noise_length": 128, "sample_rate": 44100,
              "runs": [{"wav_length": 44100, "latent_length": 15, "duration_seconds": 1,
                        "inference_ms": ms} for ms in (9999, 80, 100)],
              "real_input_stages": {name: {"relative_l2": .01} for name in ("duration_predictor", "text_encoder", "vocoder")},
              "flow_end_to_end": {"relative_l2": .02}, "spectral_convergence": .03}
    for name in ("baseline", "candidate"):
        out = directory / "en-short" / name
        out.mkdir(parents=True)
        run.write_json(out / "report.json", report)
    run.write_json(directory / "summary.json", {"completed_utc": "done", "cases": {"en-short": {
        "baseline": {"exit_code": 0}, "candidate": {"exit_code": 0, "bit_exact_with_baseline": True, "flow_bit_exact_with_baseline": True}}}})
    return directory / "en-short/candidate/report.json"


def test_reads_raw_warm_measurements_and_excludes_cold_load(protocol, tmp_path):
    write_batch(tmp_path, protocol)
    result = run.read_batch(tmp_path, protocol, 42, True)["en-short/seed42"]
    assert result["candidate"]["median_ms"] == 90
    assert result["candidate"]["p90_ms"] == 98
    assert result["exact"]


@pytest.mark.parametrize("problem", ["nan", "negative", "duration", "seed", "steps", "tracing", "missing_metric", "short_batch"])
def test_corrupt_or_mismatched_measurements_are_rejected(protocol, tmp_path, problem):
    path = write_batch(tmp_path, protocol)
    report = run.read_json(path)
    if problem == "nan":
        report["runs"][-1]["inference_ms"] = float("nan")
    elif problem == "negative":
        report["runs"][-1]["inference_ms"] = -1
    elif problem == "duration":
        for r in report["runs"]:
            r["duration_seconds"] = 2
    elif problem in ("seed", "steps"):
        report[problem] += 1
    elif problem == "tracing":
        report["runs"][-1]["trace_events"] = []
    elif problem == "missing_metric":
        del report["flow_end_to_end"]
    else:
        report["runs"].pop()
    path.write_text(json.dumps(report))
    with pytest.raises((ValueError, KeyError)):
        run.read_batch(tmp_path, protocol, 42, True)


def test_frozen_inputs_cannot_be_changed(tmp_path):
    path = tmp_path / "evaluation.py"
    path.write_text("original")
    frozen = run.fingerprint([path])
    run.verify_files(frozen)
    path.write_text("different")
    with pytest.raises(ValueError, match="Frozen input changed"):
        run.verify_files(frozen)


def test_routes_separate_runners_and_reverses_timing_order(protocol, tmp_path, monkeypatch):
    campaign = {"protocol": protocol, "baseline": {"models": "base"}, "baseline_runner": "old.exe"}
    calls = []

    def process(command, log, timeout):
        calls.append((command, timeout))
        write_batch(log.parent, protocol)
        return 0

    monkeypatch.setattr(run, "run_process", process)
    run.qualify(campaign, "candidate", "new.exe", tmp_path / "batch", 42, 1, False)
    command, timeout = calls[0]
    assert "baseline=old.exe" in command and "candidate=new.exe" in command
    assert "--timing-only" in command and "--reverse-order" in command
    assert timeout == protocol["timing_timeout_seconds"]


def test_completed_quality_failure_is_not_mistaken_for_a_crash(protocol, tmp_path, monkeypatch):
    campaign = {"protocol": protocol, "baseline": {"models": "base"}, "baseline_runner": "old.exe"}

    def process(command, log, timeout):
        path = write_batch(log.parent, protocol)
        for name in ("baseline", "candidate"):
            path = log.parent / "en-short" / name / "report.json"
            report = run.read_json(path)
            report["flow_end_to_end"]["relative_l2"] = .08
            run.write_json(path, report)
        return 1

    monkeypatch.setattr(run, "run_process", process)
    rows = run.qualify(campaign, "candidate", "new.exe", tmp_path / "batch", 42, 0, True)
    result = run.decide(rows, [measurements()] * 3, protocol)
    assert result["status"] == "keep" and result["absolute_qualification"] == "partial"


def test_process_timeout_stops_child(tmp_path):
    with pytest.raises(subprocess.TimeoutExpired):
        run.run_process([sys.executable, "-c", "import time; time.sleep(30)"], tmp_path / "timeout.log", .2)


def test_model_preflight_rejects_cpu_fallback_and_changed_weights(protocol, tmp_path, monkeypatch):
    program = tmp_path / "vector_estimator.pte"
    program.write_bytes(b"program")
    metadata = {"text_length": 128, "latent_length": 128, "cpu_compute": [], "delegates": 1,
                "target_soc": 88, "htp_arch": 81, "precision": "fp16", "model_sha256": "original"}
    run.write_json(program.with_suffix(".json"), metadata)
    config = {"bundle": None, "components": [{"path": str(program), "pte_sha256": run.digest(program), "metadata": metadata}]}
    monkeypatch.setattr(run, "model_configuration", lambda _: copy.deepcopy(config))
    baseline = run.model_snapshot(tmp_path, protocol)
    metadata["cpu_compute"] = ["aten.add"]
    with pytest.raises(ValueError, match="out-of-track"):
        run.model_snapshot(tmp_path, protocol)
    metadata["cpu_compute"] = []
    metadata["model_sha256"] = "different"
    with pytest.raises(ValueError, match="Source weights"):
        run.model_snapshot(tmp_path, protocol, baseline)
