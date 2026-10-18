"""Fixed comparison of the composed Superfast preset with the vocoder winner.

Run freeze, timing, quality, asr, report in order. Hardware ownership is checked;
timing never overlaps ASR or reference work. Every raw timing is retained.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import html
import json
import math
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import wave

from build_superfast import ROOT, RESEARCH, FRONTEND, VOCODER, read, sha

OUT = ROOT / "artifacts/superfast-oct18"
PRESET = ROOT / "artifacts/superfast"
LEASE = RESEARCH / "o16w-resource-owner.json"
RUNNER = RESEARCH / "worktree-oct16/artifacts/campaign-oct16/baseline-runner/supertonic_qnn.exe"
VOCODER_OUT = RESEARCH / "worktree-vocoder-oct16/artifacts/vocoder-head-oct16"
FRONTEND_OUT = RESEARCH / "worktree-frontend-layout-oct16/artifacts/frontend-layout-oct16/campaign"
ASR_ROOT = RESEARCH / "worktree-mlp-oct16/artifacts/asr"
STAGES = ("duration_predictor", "text_encoder", "vector_estimator", "vocoder")


def now():
    return datetime.now(timezone.utc).isoformat()


def save(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def owner():
    lease = read(LEASE)
    if not os.environ.get("CODEX_THREAD_ID") or lease["owner_thread"] != os.environ["CODEX_THREAD_ID"] or lease["owner_task"] != "superfast-oct18":
        raise RuntimeError("This campaign requires exclusive superfast-oct18 resource ownership")


def verify():
    owner()
    for path, digest in read(OUT / "manifest.json")["identities"].items():
        if sha(path) != digest:
            raise ValueError(f"Frozen input changed: {path}")


def reference_source(case, role):
    if role == "baseline":
        return VOCODER_OUT / "real-inputs" / f"{case['id']}-{case['seed']}" / "baseline"
    return FRONTEND_OUT / "quality/pack8-norm-tail" / f"seed-{case['seed']}" / case["id"] / "candidate"


def freeze():
    owner()
    if (OUT / "manifest.json").exists():
        raise FileExistsError("Campaign is already frozen")
    from qnn_runtime import model_configuration
    handoff = read(RESEARCH / "worktree-vocoder-oct16/research/vocoder_handoff.json")
    protocol = {"created_utc": now(), "candidate": str(PRESET), "baseline": str(VOCODER),
                "steps": 8, "speed": 1.05, "noise_capacity": 128, "text_capacity": 128, "audio_bucket": 96,
                "timing_cases": [{**c, "seed": 42} for c in handoff["timing_cases"]],
                "quality_cases": handoff["quality_cases"], "rounds": 3, "requests": 16, "discard": 1,
                "order": "baseline-before, candidate, baseline-after; case order reversed in odd rounds",
                "control": "one unchanged-baseline bracket per timing case before candidate measurements",
                "reruns": "none; all attempts retained",
                "gain_rule": "Positive reduction in each round and aggregate reduction above maximum local/inter-round baseline drift. Descriptive repeatability heuristic, not statistical proof.",
                "quality": "Six paired cases; finite/non-silent/clipping checks, common-prefix spectral diagnostics without time warping, full WAV ASR. No strict numerical gate; human listening pending.",
                "reference": "Reuse original FP32 text/flow/vocoder teachers only after checking complete raw inputs, HTP duration, latent length, crop and asset identities. Each role uses its own HTP duration; differences are reported.",
                "timing_scope": "Warm native inference excludes model loading, Python preprocessing and tensor file IO; native request includes tensor IO.",
                "asr": {"model": "Whisper-small", "device": "cpu", "compute_type": "int8_float32", "threads": 2},
                "target_rtf": .01}
    save(OUT / "protocol.json", protocol)
    paths = {OUT / "protocol.json", PRESET / "bundle.json", PRESET / "provenance.json", VOCODER / "bundle.json", RUNNER}
    configurations = {}
    for role, bundle in (("candidate", PRESET), ("baseline", VOCODER)):
        configurations[role] = model_configuration(bundle)
        for component in configurations[role]["components"]:
            p = Path(component["path"])
            paths.update((p, p.with_suffix(".json")))
        data = read(bundle / "bundle.json")
        paths.add((bundle / data["base"] / "vector_estimator.json").resolve())
    paths.update((ROOT / "assets").rglob("*.json"))
    paths.update((ROOT / "assets/onnx").glob("*.onnx"))
    paths.update(RUNNER.parent.glob("*.dll"))
    sdk = Path("C:/Qualcomm/AIStack/QAIRT/2.50.40.260831")
    for folder in ("lib/aarch64-windows-msvc", "lib/hexagon-v81/unsigned"):
        paths.update(p for p in (sdk / folder).glob("*") if p.is_file())
    paths.update(Path("C:/Windows/System32/DriverStore/FileRepository").glob("qcnspmcdm8480.inf_arm64_*/libcdsprpc.dll"))
    sources = [ROOT / "scripts" / name for name in ("build_superfast.py", "measure_superfast.py", "synthesize.py", "qnn_runtime.py")]
    sources += [RESEARCH / "worktree-mlp-oct16/scripts/evaluate_intelligibility.py"]
    for p in sources:
        paths.add(p)
        target = OUT / "sources" / p.name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(p, target)
    # The historical manifests bind teacher generation to the same ONNX assets.
    for history in (FRONTEND_OUT / "manifest.json", VOCODER_OUT / "manifest.json"):
        paths.add(history)
        historical_assets = {str(Path(p).resolve()): digest for p, digest in read(history)["identities"].items()
                             if Path(p).suffix == ".onnx"}
        for name in STAGES:
            asset = (ROOT / "assets/onnx" / f"{name}.onnx").resolve()
            assert historical_assets[str(asset)] == sha(asset), f"Teacher model identity mismatch: {name}"
    for case in protocol["quality_cases"]:
        for role in ("baseline", "candidate"):
            source = reference_source(case, role)
            paths.update(source / name for name in ("onnx_reference.wav", "report.json", "duration.raw"))
            paths.update((source / "inputs").glob("*.raw"))
    identities = {str(p.resolve()): sha(p) for p in sorted(paths)}
    save(OUT / "manifest.json", {"created_utc": now(), "identities": identities, "configurations": configurations,
         "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
         "power_scheme": subprocess.check_output(["powercfg", "/getactivescheme"], text=True),
         "python": sys.version, "executable": sys.executable})
    print(f"Frozen {len(identities)} identities", flush=True)


def synthesis(bundle, case, dest, repeat):
    owner()
    dest.mkdir(parents=True, exist_ok=False)
    argv = [sys.executable, ROOT / "scripts/synthesize.py", "--models", bundle, "--runner", RUNNER,
            "--text", case["text"], "--lang", case["lang"], "--voice", case["voice"], "--seed", str(case["seed"]),
            "--steps", "8", "--speed", "1.05", "--repeat", str(repeat), "--output", dest]
    journal = {"argv": list(map(str, argv)), "cwd": str(ROOT), "started_utc": now()}
    save(dest / "command.json", journal)
    env = os.environ.copy()
    env.update(OMP_NUM_THREADS="4", MKL_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4", PYTHONIOENCODING="utf-8")
    with (dest / "run.log").open("w", encoding="utf-8") as log:
        process = subprocess.Popen(journal["argv"], cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        journal["pid"] = process.pid
        save(dest / "command.json", journal)
        journal["exit_code"] = process.wait()
    journal["completed_utc"] = now()
    save(dest / "command.json", journal)
    if journal["exit_code"]:
        raise RuntimeError(f"Synthesis failed: {dest / 'run.log'}")
    report = read(dest / "report.json")
    assert all(len(r["vector_estimator_ms"]) == 8 and r["bucket_length"] == 96 and not r.get("trace_events") for r in report["runs"])
    assert report["audio"]["clipped_sample_fraction"] <= .01
    return report


def bracket(case, round_index, dest, candidate):
    import numpy as np
    reports = {name: synthesis(candidate if name == "candidate" else VOCODER, case, dest / name, 16)
               for name in ("baseline-before", "candidate", "baseline-after")}
    assert sha(dest / "baseline-before/audio.raw") == sha(dest / "baseline-after/audio.raw")
    if candidate == VOCODER:
        assert sha(dest / "candidate/audio.raw") == sha(dest / "baseline-before/audio.raw")
    a, b, z = [reports[name] for name in ("baseline-before", "candidate", "baseline-after")]
    aw, bw, zw = [x["warm_summary"] for x in (a, b, z)]
    before, after = aw["synthesis_ms_median"], zw["synthesis_ms_median"]
    mean = (before + after) / 2
    seconds = b["runs"][-1]["wav_length"] / b["sample_rate"]
    base_seconds = a["runs"][-1]["wav_length"] / a["sample_rate"]
    row = {"round": round_index, "case": case["id"], "baseline_before_ms": before, "baseline_after_ms": after,
           "baseline_ms": mean, "candidate_ms": bw["synthesis_ms_median"], "ratio": bw["synthesis_ms_median"] / mean,
           "bracket_drift": abs(after / before - 1), "candidate_p90_ms": bw["synthesis_ms_p90"],
           "baseline_p90_ms": (aw["synthesis_ms_p90"] + zw["synthesis_ms_p90"]) / 2,
           "candidate_stage_ms": bw["stage_ms_median"],
           "baseline_stage_ms": {s: (aw["stage_ms_median"][s] + zw["stage_ms_median"][s]) / 2 for s in STAGES},
           "audio_seconds": seconds, "baseline_audio_seconds": base_seconds,
           "candidate_request_ms": float(np.median([r["request_ms"] for r in b["runs"][1:]])),
           "candidate_request_p90_ms": float(np.percentile([r["request_ms"] for r in b["runs"][1:]], 90)),
           "baseline_request_ms": float(np.mean([np.median([r["request_ms"] for r in rep["runs"][1:]]) for rep in (a, z)]))}
    row.update(inference_rtf=row["candidate_ms"] / (1000 * seconds), baseline_duration_rtf=row["candidate_ms"] / (1000 * base_seconds),
               request_rtf=row["candidate_request_ms"] / (1000 * seconds))
    print(json.dumps(row), flush=True)
    return row


def timing():
    verify()
    p = read(OUT / "protocol.json")
    assert not (OUT / "timing").exists() and not (OUT / "control").exists()
    control = [bracket(c, 0, OUT / "control" / c["id"], VOCODER) for c in p["timing_cases"]]
    save(OUT / "control.json", control)
    control_ratio = math.exp(statistics.mean(math.log(r["ratio"]) for r in control))
    assert abs(control_ratio - 1) <= max(.03, max(r["bracket_drift"] for r in control)), "Unexpected unchanged-control difference"
    rows = []
    for i in range(p["rounds"]):
        for case in p["timing_cases"][::1 if i % 2 == 0 else -1]:
            rows.append(bracket(case, i, OUT / "timing" / f"round-{i}" / case["id"], PRESET))
            save(OUT / "timing.json", {"complete": False, "rows": rows})
    verify()
    save(OUT / "timing.json", {"complete": True, "completed_utc": now(), "rows": rows})


def quality():
    verify()
    import numpy as np
    import soundfile as sf
    from qnn_runtime import metrics
    from synthesize import spectrum
    rows = []
    for case in read(OUT / "protocol.json")["quality_cases"]:
        dest = OUT / "quality" / f"{case['id']}-{case['seed']}"
        records = {}
        for role, bundle in (("baseline", VOCODER), ("candidate", PRESET)):
            folder = dest / role
            report = synthesis(bundle, case, folder, 2)
            source = reference_source(case, role)
            historical = read(source / "report.json")
            for key in ("text", "lang", "voice", "seed", "steps", "speed", "sample_rate", "noise_length"):
                assert report[key] == historical[key], key
            assert report["runs"][-1]["wav_length"] == historical["runs"][-1]["wav_length"]
            assert report["runs"][-1]["latent_length"] == historical["runs"][-1]["latent_length"]
            for name in ("text_ids", "text_mask", "noise", "style_ttl", "style_dp"):
                assert sha(folder / f"inputs/{name}.raw") == sha(source / f"inputs/{name}.raw"), name
            assert sha(folder / "duration.raw") == sha(source / "duration.raw")
            shutil.copyfile(source / "onnx_reference.wav", folder / "onnx_reference.wav")
            actual, sr = sf.read(folder / "supertonic_htp.wav", dtype="float32")
            teacher, tsr = sf.read(folder / "onnx_reference.wav", dtype="float32")
            assert sr == tsr and actual.shape == teacher.shape and np.isfinite(actual).all()
            records[role] = {"report": str(folder / "report.json"), "audio_seconds": len(actual) / sr,
                             "teacher_source": str(source / "onnx_reference.wav"), "teacher_sha256": sha(folder / "onnx_reference.wav"),
                             "waveform_vs_fp32": metrics(actual, teacher), "spectral_vs_fp32": float(np.linalg.norm(spectrum(actual) - spectrum(teacher)) / np.linalg.norm(spectrum(teacher)))}
        a, _ = sf.read(dest / "baseline/supertonic_htp.wav", dtype="float32")
        b, _ = sf.read(dest / "candidate/supertonic_htp.wav", dtype="float32")
        count = min(a.size, b.size)
        row = {"case": case, "path": str(dest), "roles": records, "duration_delta_percent": 100 * (b.size / a.size - 1),
               "paired_common_prefix_waveform": metrics(b[:count], a[:count]),
               "paired_common_prefix_spectral": float(np.linalg.norm(spectrum(b[:count]) - spectrum(a[:count])) / np.linalg.norm(spectrum(a[:count])))}
        rows.append(row)
        save(OUT / "quality.json", {"complete": False, "cases": rows})
    verify()
    save(OUT / "quality.json", {"complete": True, "cases": rows, "human_listening": "pending", "numeric_audio": "PCM16 WAV common prefix without alignment/time warping"})


def asr():
    verify()
    assert read(OUT / "timing.json")["complete"] and read(OUT / "quality.json")["complete"]
    from dataclasses import asdict
    sys.path.insert(0, str(RESEARCH / "worktree-mlp-oct16/scripts"))
    import evaluate_intelligibility as e
    os.environ.update(HF_HUB_OFFLINE="1", HF_HUB_DISABLE_TELEMETRY="1", TRANSFORMERS_OFFLINE="1")
    from faster_whisper import WhisperModel
    import ctranslate2
    identity, packages = e.verify_model(ASR_ROOT / "model"), e.package_versions()
    ctranslate2.set_random_seed(0)
    model = WhisperModel(str(ASR_ROOT / "model"), device="cpu", compute_type="int8_float32", cpu_threads=2, num_workers=1, local_files_only=True)
    assert model.model.compute_type == "int8_float32" and model.model.is_multilingual
    assert not (OUT / "asr.json").exists()
    result = {"model": identity, "packages": packages, "decode": e.DECODE, "cases": [], "complete": False}
    cache = {}
    for row in read(OUT / "quality.json")["cases"]:
        case = row["case"]
        record = {"case": case, "roles": {}}
        for role, folder, filename in (("baseline", "baseline", "supertonic_htp.wav"), ("candidate", "candidate", "supertonic_htp.wav"),
                                        ("baseline_fp32", "baseline", "onnx_reference.wav"), ("candidate_fp32", "candidate", "onnx_reference.wav")):
            path = Path(row["path"]) / folder / filename
            key = (sha(path), case["lang"])
            if key not in cache:
                segments, info = model.transcribe(str(path), language=case["lang"], **e.DECODE)
                segments = [asdict(s) for s in segments]
                text = "".join(s["text"] for s in segments).strip()
                assert e.normalize(text), "Empty transcript"
                cache[key] = {"text": text, "segments": segments, "info": asdict(info)}
            decoded = cache[key]
            record["roles"][role] = {"path": str(path), "sha256": key[0], **decoded,
                                     "raw": e.score(case["text"], decoded["text"]),
                                     "normalized": e.score(e.normalize(case["text"]), e.normalize(decoded["text"]))}
        result["cases"].append(record)
        save(OUT / "asr.json", result)
        print("ASR complete", case["id"], case["seed"], flush=True)
    aggregates = {}
    for role in result["cases"][0]["roles"]:
        aggregates[role] = {}
        for metric in ("wer", "cer"):
            errors = sum(r["roles"][role]["normalized"][metric]["errors"] for r in result["cases"])
            units = sum(r["roles"][role]["normalized"][metric]["reference_units"] for r in result["cases"])
            aggregates[role][metric] = {"errors": errors, "units": units, "rate": errors / units}
    result.update(aggregates=aggregates, complete=True, completed_utc=now())
    save(OUT / "asr.json", result)
    verify()


def report():
    verify()
    timing, quality, asr_result = [read(OUT / n) for n in ("timing.json", "quality.json", "asr.json")]
    assert all(x["complete"] for x in (timing, quality, asr_result))
    rows = timing["rows"]
    gm = lambda values: math.exp(statistics.mean(math.log(x) for x in values))
    round_gains = [100 * (1 - gm(r["ratio"] for r in rows if r["round"] == i)) for i in range(3)]
    gain = 100 * (1 - gm(r["ratio"] for r in rows))
    drift = 100 * max(r["bracket_drift"] for r in rows)
    cases = {}
    for name in sorted({r["case"] for r in rows}):
        selected = [r for r in rows if r["case"] == name]
        inter = 100 * (max(r["baseline_ms"] for r in selected) / min(r["baseline_ms"] for r in selected) - 1)
        drift = max(drift, inter)
        cases[name] = {key: statistics.median(r[key] for r in selected) for key in
                      ("candidate_ms", "baseline_ms", "candidate_p90_ms", "baseline_p90_ms", "candidate_request_ms", "baseline_request_ms",
                       "audio_seconds", "baseline_audio_seconds", "inference_rtf", "baseline_duration_rtf", "request_rtf")}
        cases[name]["stage_ms"] = {s: {role: statistics.median(r[role + "_stage_ms"][s] for r in selected) for role in ("baseline", "candidate")} for s in STAGES}
        cases[name]["baseline_interround_drift_percent"] = inter
    summary = {"completed_utc": now(), "scope": read(OUT / "protocol.json"), "cases": cases, "gain_percent": gain,
               "round_gains_percent": round_gains, "max_drift_percent": drift,
               "repeatable_gain": all(g > 0 for g in round_gains) and gain > drift,
               "target_rtf_met_every_case_round": all(r["inference_rtf"] <= .01 and r["baseline_duration_rtf"] <= .01 for r in rows),
               "asr": asr_result["aggregates"], "max_paired_spectral": max(r["paired_common_prefix_spectral"] for r in quality["cases"]),
               "max_candidate_spectral_vs_fp32": max(r["roles"]["candidate"]["spectral_vs_fp32"] for r in quality["cases"]),
               "max_duration_delta_percent": max(abs(r["duration_delta_percent"]) for r in quality["cases"]),
               "human_listening": "pending", "source_manifest_sha256": sha(OUT / "manifest.json")}
    save(OUT / "summary.json", summary)
    save(ROOT / "benchmarks/superfast-2026-10-18.json", {**summary, "timing_rounds": rows, "quality": quality})
    html_rows = []
    for r in quality["cases"]:
        c = r["case"]
        cells = []
        for label, folder, filename in (("Vocoder winner / original frontends", "baseline", "supertonic_htp.wav"),
                                       ("Superfast / combined winners", "candidate", "supertonic_htp.wav"),
                                       ("Original FP32 / superfast duration", "candidate", "onnx_reference.wav")):
            p = Path(r["path"]) / folder / filename
            rel = Path(os.path.relpath(p, OUT)).as_posix()
            cells.append(f'<td>{html.escape(label)}<br><audio controls preload="none" src="{html.escape(rel)}"></audio></td>')
        html_rows.append(f'<tr><td>{html.escape(c["id"])} / {c["voice"]} / seed {c["seed"]}<br>{html.escape(c["text"])}</td>{"".join(cells)}</tr>')
    (OUT / "listening.html").write_text('<!doctype html><html lang="en"><meta charset="utf-8"><title>Superfast paired listening</title><style>body{font:16px/1.5 system-ui;margin:30px}td{vertical-align:top;padding:14px;border-bottom:1px solid #ddd}audio{width:260px}table{border-collapse:collapse}</style><h1>Superfast: eight evaluations</h1><p>Optimized frontends + full28 packed flow with shared CFG prefix + combined vocoder. Baseline uses the same flow/vocoder with original frontends. FP32 uses original text encoding, flow and vocoder with the superfast HTP duration and identical noise. Duration differences are retained. Human listening is pending; numerical and ASR scores are diagnostics.</p><p><a href="summary.json">Timing and quality summary</a> | <a href="asr.json">Full ASR transcripts</a></p><table>'+''.join(html_rows)+'</table></html>', encoding="utf-8")
    audios = []
    for p in sorted((OUT / "quality").rglob("*.wav")):
        with wave.open(str(p), "rb") as wav:
            assert wav.getnchannels() == 1 and wav.getsampwidth() == 2 and wav.getframerate() == 44100
            count = wav.getnframes()
            assert len(wav.readframes(count)) == count * 2
        audios.append({"path": str(p), "sha256": sha(p), "frames": count})
    assert len(audios) == 24
    commands = [read(p) for p in OUT.rglob("command.json")]
    assert all(c.get("exit_code") == 0 and c.get("completed_utc") for c in commands)
    verify()
    hashes = {str(p.relative_to(OUT)): sha(p) for p in sorted(OUT.rglob("*")) if p.is_file() and p.name not in ("artifact-hashes.json", "audit.json")}
    save(OUT / "artifact-hashes.json", hashes)
    save(OUT / "audit.json", {"completed_utc": now(), "all_frozen_identities_verified": True, "audio": audios,
                              "artifact_count": len(hashes), "terminal_native_commands": len(commands), "warm_timing_requests": 24 * 15})
    print(json.dumps(summary, ensure_ascii=True), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("freeze", "timing", "quality", "asr", "report"))
    phase = parser.parse_args().phase
    globals()[phase]()
