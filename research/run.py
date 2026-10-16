"""Controlled TTS inference experiments. See program.md before starting a campaign."""

import argparse
from contextlib import contextmanager
import csv
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import re
import shutil
import signal
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from qnn_runtime import DEFAULT_SDK, model_configuration

EXECUTABLE = ROOT / "build/runner/Release/supertonic_qnn.exe"
PROTECTED = ["research/run.py", "research/program.md", "scripts/qualify_optimizations.py",
             "scripts/synthesize.py", "scripts/qnn_runtime.py", "scripts/optimization_cases.json",
             "external/supertonic/py/helper.py", "requirements.txt"]


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def fingerprint(paths):
    return {str(Path(p).resolve()): digest(p) for p in sorted(set(map(Path, paths)))}


def verify_files(expected):
    for path, sha in expected.items():
        if not Path(path).is_file() or digest(path) != sha:
            raise ValueError(f"Frozen input changed: {path}; start a new campaign after review")


def validate_protocol(p):
    if p["version"] != 1 or p["track"] != "exact-execution":
        raise ValueError("Only the exact-execution v1 protocol is implemented")
    available = {c["id"] for c in read_json(ROOT / "scripts/optimization_cases.json")["qualification"]}
    if not p["cases"] or len(set(p["cases"])) != len(p["cases"]) or set(p["cases"]) - available:
        raise ValueError("Cases must be nonempty, unique qualification case IDs")
    if not p["seeds"] or len(set(p["seeds"])) != len(p["seeds"]) or any(type(s) is not int or s < 0 for s in p["seeds"]):
        raise ValueError("Seeds must be unique nonnegative integers")
    for key, minimum in (("steps", 1), ("quality_requests", 3), ("timing_requests", 3), ("timing_rounds", 3)):
        if type(p[key]) is not int or p[key] < minimum:
            raise ValueError(f"Invalid {key}")
    if p["steps"] > 100:
        raise ValueError("At most 100 flow steps")
    for key in ("speed", "minimum_gain", "maximum_case_regression", "maximum_p90_regression",
                "maximum_program_size_ratio", "stage_relative_l2_max", "final_latent_relative_l2_max",
                "spectral_convergence_max", "build_timeout_seconds", "quality_timeout_seconds", "timing_timeout_seconds"):
        if not math.isfinite(p[key]) or p[key] <= 0:
            raise ValueError(f"Invalid {key}")
    if not math.isfinite(p["settle_seconds"]) or not 0 <= p["settle_seconds"] <= 60:
        raise ValueError("settle_seconds must be between 0 and 60")


def environment():
    power = subprocess.run(["powercfg", "/getactivescheme"], capture_output=True, text=True, check=True).stdout.strip() if os.name == "nt" else "unavailable"
    return {"machine": platform.node(), "platform": platform.platform(), "python": sys.version,
            "python_executable": sys.executable, "power_scheme": power,
            "packages": {name: importlib.metadata.version(name) for name in ("numpy", "onnxruntime", "soundfile")}}


def source_state(output):
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    status = subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True)
    patch = subprocess.check_output(["git", "diff", "--binary", "HEAD"], cwd=ROOT)
    (output / "working-tree.patch").write_bytes(patch)
    # Include untracked research/implementation files, which git diff does not contain.
    paths = [p for folder in ("scripts", "native", "research") for p in (ROOT / folder).rglob("*")
             if p.is_file() and p.suffix in (".py", ".cpp", ".h", ".json", ".md", ".ps1", ".txt")]
    for path in paths:
        target = output / "source" / path.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
    return {"commit": commit, "status": status, "files": fingerprint(paths)}


def runner_files(executable):
    executable = Path(executable).resolve()
    return fingerprint([executable, *executable.parent.glob("*.dll")])


def model_snapshot(models, protocol, reference=None):
    models = Path(models).resolve()
    config = model_configuration(models)
    components = config["components"]
    base = models / config["bundle"]["base"] if config["bundle"] else models
    base_meta = read_json(base / "vector_estimator.json")
    shapes = {"text_length": base_meta["text_length"], "noise_length": base_meta["latent_length"]}
    weights = {}
    total_bytes = 0
    extra = [base / "vector_estimator.json"]
    if config["bundle"]:
        extra.append(models / "bundle.json")
        if config["bundle"].get("flow_total_steps", protocol["steps"]) != protocol["steps"]:
            raise ValueError("Bundle step schedule differs from the protocol")
    for item in components:
        meta = item["metadata"]
        component = Path(item["path"]).stem
        if (meta["cpu_compute"] or meta["delegates"] != 1 or meta.get("profile_level", "off") != "off"
                or meta["target_soc"] != 88 or meta["htp_arch"] != 81 or meta["precision"] != "fp16"
                or meta.get("gelu", "exact") != "exact"):
            raise ValueError(f"Non-production or out-of-track program: {item['path']}")
        if meta.get("flow_total_steps", protocol["steps"]) != protocol["steps"]:
            raise ValueError("Compiled flow schedule differs from the protocol")
        if component in weights and weights[component] != meta["model_sha256"]:
            raise ValueError("Buckets use different source weights")
        weights[component] = meta["model_sha256"]
        total_bytes += Path(item["path"]).stat().st_size
    if reference and (weights != reference["weights"] or shapes != reference["shapes"]):
        raise ValueError("Source weights or input/noise capacity differ from the baseline")
    return {"models": str(models), "configuration": config, "extra_files": fingerprint(extra),
            "weights": weights, "shapes": shapes, "program_bytes": total_bytes}


def initialize(args):
    p = read_json(args.protocol)
    validate_protocol(p)
    baseline = model_snapshot(args.baseline, p)
    # Actual reference data are hashed, not merely the manifest that describes them.
    asset_files = [ROOT / "assets/manifest.json"]
    asset_files += [ROOT / "assets" / f["path"] for f in read_json(asset_files[0])["files"]]
    sdk_files = [DEFAULT_SDK / "lib/aarch64-windows-msvc" / name for name in
                 ("QnnHtp.dll", "QnnSystem.dll", "QnnHtpV81Stub.dll", "QnnHtpPrepare.dll")]
    sdk_files += list((DEFAULT_SDK / "lib/hexagon-v81/unsigned").glob("libQnn*.so"))
    sdk_files += list(Path(r"C:\Windows\System32\DriverStore\FileRepository").glob("qcnspmcdm8480.inf_arm64_*/libcdsprpc.dll"))
    frozen = fingerprint([*(ROOT / f for f in PROTECTED), args.protocol, *asset_files, *sdk_files])
    args.campaign.mkdir(parents=True, exist_ok=False)
    native = args.campaign / "baseline-runner"
    native.mkdir()
    for file in runner_files(args.runner):
        shutil.copy2(file, native / Path(file).name)
    write_json(args.campaign / "campaign.json", {
        "created_utc": datetime.now(timezone.utc).isoformat(), "protocol": p,
        "frozen_files": frozen, "baseline": baseline,
        "baseline_runner": str((native / args.runner.name).resolve()),
        "baseline_runner_files": runner_files(native / args.runner.name),
        "environment": environment(), "source": source_state(args.campaign),
    })
    print(f"Initialized {args.campaign.resolve()}; no experiments started")


def run_process(command, log_path, timeout):
    """Kill the entire experiment process tree on timeout or interruption."""
    with log_path.open("w", encoding="utf-8") as log:
        proc = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                start_new_session=os.name != "nt")
        try:
            return proc.wait(timeout=timeout)
        except BaseException:
            if os.name == "nt":
                subprocess.run(["taskkill", "/PID", str(proc.pid), "/T", "/F"], stdout=log, stderr=log)
            else:
                os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()
            raise


@contextmanager
def device_lock():
    path = ROOT / "artifacts/research/device.lock"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        stream.write(str(os.getpid()))
    try:
        yield
    finally:
        path.unlink()


def number(value, positive=False):
    if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or value < 0 or (positive and value == 0):
        raise ValueError(f"Invalid measurement: {value!r}")
    return value


def percentile90(values):
    values = sorted(values)
    index = .9 * (len(values) - 1)
    lo = math.floor(index)
    return values[lo] + (values[math.ceil(index)] - values[lo]) * (index - lo)


def read_batch(path, p, seed, quality):
    summary = read_json(path / "summary.json")
    if not summary.get("completed_utc") or set(summary["cases"]) != set(p["cases"]):
        raise ValueError("Incomplete qualification batch")
    expected_cases = {c["id"]: c for c in read_json(ROOT / "scripts/optimization_cases.json")["qualification"]}
    rows = {}
    for case_id, variants in summary["cases"].items():
        if set(variants) != {"baseline", "candidate"}:
            raise ValueError("Missing or unexpected variant")
        row = {"case": case_id, "seed": seed, "absolute_failures": {}, "exact": True}
        duration_shapes = []
        for name, result in variants.items():
            if result["exit_code"] != 0:
                raise ValueError(f"Synthesis failed for {case_id}/{name}; see batch logs")
            report = read_json(path / case_id / name / "report.json")
            expected = expected_cases[case_id]
            for key, value in {"seed": seed, "steps": p["steps"], "speed": p["speed"],
                               **{key: expected[key] for key in ("text", "lang", "voice")}}.items():
                if report[key] != value:
                    raise ValueError(f"Input changed: {case_id}/{name}/{key}")
            runs = report["runs"]
            count = p["quality_requests"] if quality else p["timing_requests"]
            if len(runs) != count:
                raise ValueError("Incomplete request count")
            shapes = {(r["wav_length"], r["latent_length"], number(r["duration_seconds"], True)) for r in runs}
            if len(shapes) != 1 or any("trace_events" in r for r in runs):
                raise ValueError("Duration changed across repetitions or tracing was enabled")
            duration_shapes.append((shapes.pop(), report["noise_length"], report["sample_rate"]))
            timings = [number(r["inference_ms"], True) for r in runs[1:]]
            row[name] = {"median_ms": statistics.median(timings), "p90_ms": percentile90(timings),
                         "rtf": statistics.median(timings) / (runs[-1]["duration_seconds"] * 1000)}
            if quality:
                measurements = {key: (number(report["real_input_stages"][key]["relative_l2"]), p["stage_relative_l2_max"])
                                for key in ("duration_predictor", "text_encoder", "vocoder")}
                measurements["final_latent"] = (number(report["flow_end_to_end"]["relative_l2"]), p["final_latent_relative_l2_max"])
                measurements["spectrum"] = (number(report["spectral_convergence"]), p["spectral_convergence_max"])
                row["absolute_failures"][name] = {k: v for k, (v, limit) in measurements.items() if v > limit}
        if duration_shapes[0] != duration_shapes[1]:
            raise ValueError("Candidate changed audio duration, valid length, sample rate or input noise capacity")
        row["exact"] = (variants["candidate"].get("bit_exact_with_baseline") is True
                        and variants["candidate"].get("flow_bit_exact_with_baseline") is True)
        rows[f"{case_id}/seed{seed}"] = row
    return rows


def geometric_mean(values):
    return math.exp(statistics.mean(math.log(v) for v in values))


def decide(quality, rounds, p):
    """Conservative repeatability screen, not a statistical confidence interval."""
    if set(quality) != {f"{case}/seed{seed}" for case in p["cases"] for seed in p["seeds"]}:
        raise ValueError("Incomplete quality cases or seeds")
    failures = {key: row["absolute_failures"] for key, row in quality.items()
                if any(row["absolute_failures"].values())}
    result = {"absolute_failures": failures, "absolute_qualification": "partial" if failures else "pass"}
    if any(not r["exact"] for r in quality.values()):
        return {**result, "status": "discard", "reason": "Audio or final latents differ in the exact-execution track"}
    for row in quality.values():
        base, candidate = row["absolute_failures"]["baseline"], row["absolute_failures"]["candidate"]
        if any(k not in base or v > base[k] + 1e-9 for k, v in candidate.items()):
            return {**result, "status": "discard", "reason": "New or worsened absolute ONNX failure"}
    if len(rounds) != p["timing_rounds"] or any(set(r) != set(quality) for r in rounds):
        raise ValueError("Incomplete timing rounds")
    if any(not row["exact"] for round_ in rounds for row in round_.values()):
        return {**result, "status": "discard", "reason": "Output changed during timing"}
    gains = [1 - geometric_mean(row["candidate"]["median_ms"] / row["baseline"]["median_ms"]
                               for row in round_.values()) for round_ in rounds]
    drift = max(max(r[key]["baseline"]["median_ms"] for r in rounds) /
                min(r[key]["baseline"]["median_ms"] for r in rounds) - 1 for key in quality)
    gain = 1 - geometric_mean(1 - g for g in gains)
    case_results = {}
    exceeded = False
    consistent_regression = False
    for key in quality:
        regressions = [r[key]["candidate"]["median_ms"] / r[key]["baseline"]["median_ms"] - 1 for r in rounds]
        tails = [r[key]["candidate"]["p90_ms"] / r[key]["baseline"]["p90_ms"] - 1 for r in rounds]
        exceeded |= max(regressions) > p["maximum_case_regression"] or max(tails) > p["maximum_p90_regression"]
        consistent_regression |= min(regressions) > p["maximum_case_regression"] or min(tails) > p["maximum_p90_regression"]
        case_results[key] = {"median_regression_by_round": regressions, "p90_regression_by_round": tails}
    result.update(gain=gain, round_gains=gains, baseline_drift=drift, cases=case_results)
    if consistent_regression or max(gains) < -p["minimum_gain"]:
        return {**result, "status": "discard", "reason": "Repeatable latency regression"}
    if not exceeded and min(gains) > p["minimum_gain"] and gain > drift:
        return {**result, "status": "keep", "reason": "Gain reproduced in every round and exceeds measured baseline drift"}
    return {**result, "status": "inconclusive", "reason": "Gain is too small, order-sensitive, noisy, or violates a per-case timing guard"}


def qualify(campaign, candidate, runner, output, seed, round_index, quality):
    p = campaign["protocol"]
    output.mkdir()
    command = [sys.executable, str(ROOT / "scripts/qualify_optimizations.py"),
               "--variant", f"baseline={campaign['baseline']['models']}", "--variant", f"candidate={candidate}",
               "--runner", f"baseline={campaign['baseline_runner']}", "--runner", f"candidate={runner}",
               "--steps", str(p["steps"]), "--speed", str(p["speed"]), "--seed", str(seed),
               "--repeat", str(p["quality_requests"] if quality else p["timing_requests"]), "--output", str(output)]
    for case in p["cases"]:
        command += ["--case", case]
    if round_index % 2:
        command.append("--reverse-order")
    if not quality:
        command.append("--timing-only")
    timeout = p["quality_timeout_seconds"] if quality else p["timing_timeout_seconds"]
    rc = run_process(command, output / "batch.log", timeout)
    # Exit 1 may mean a fully recorded, inherited absolute quality failure.
    if rc not in ((0, 1) if quality else (0,)):
        raise ValueError(f"Qualification exited {rc}; see {output / 'batch.log'}")
    return read_batch(output, p, seed, quality)


def experiment(args):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", args.tag):
        raise ValueError("Tag must contain only letters, digits, hyphens or underscores")
    campaign_path = args.campaign / "campaign.json"
    campaign = read_json(campaign_path)
    campaign_sha = digest(campaign_path)
    p = campaign["protocol"]
    validate_protocol(p)
    with device_lock():
        output = args.campaign / args.tag
        output.mkdir(exist_ok=False)
        result = {"tag": args.tag, "description": args.description,
                  "started_utc": datetime.now(timezone.utc).isoformat(), "campaign_sha256": campaign_sha}
        try:
            verify_files(campaign["frozen_files"])
            verify_files(campaign["baseline_runner_files"])
            if environment() != campaign["environment"]:
                raise ValueError("Machine, power scheme or Python environment changed")
            baseline = model_snapshot(campaign["baseline"]["models"], p)
            if baseline != campaign["baseline"]:
                raise ValueError("Baseline programs or metadata changed")
            if args.build_command:
                command = read_json(args.build_command)
                if not isinstance(command, list) or not command or any(not isinstance(v, str) for v in command):
                    raise ValueError("Build command must be a JSON array of argument strings")
                result["build_command"] = command
                started = time.monotonic()
                rc = run_process(command, output / "build.log", p["build_timeout_seconds"])
                result["build_seconds"] = time.monotonic() - started
                if rc:
                    raise ValueError(f"Build exited {rc}")
            verify_files(campaign["frozen_files"])
            candidate = model_snapshot(args.candidate, p, baseline)
            result.update(candidate=candidate, candidate_runner_files=runner_files(args.runner),
                          source=source_state(output), environment=environment(), peak_memory_bytes=None)
            if candidate["program_bytes"] > baseline["program_bytes"] * p["maximum_program_size_ratio"]:
                result.update(status="discard", reason="Compiled program size exceeds the frozen budget")
            else:
                quality = {}
                for index, seed in enumerate(p["seeds"]):
                    print(f"Quality seed {seed}", flush=True)
                    quality.update(qualify(campaign, args.candidate, args.runner, output / f"quality-{seed}", seed, index, True))
                result["quality"] = quality
                # Failed correctness is sufficient to reject; don't waste device timing.
                preliminary = decide(quality, [], p) if any(not r["exact"] for r in quality.values()) else None
                if preliminary:
                    result.update(preliminary)
                else:
                    rounds = []
                    for index in range(p["timing_rounds"]):
                        round_ = {}
                        for seed in p["seeds"]:
                            time.sleep(p["settle_seconds"])
                            print(f"Timing round {index + 1}, seed {seed}", flush=True)
                            round_.update(qualify(campaign, args.candidate, args.runner, output / f"timing-{index}-{seed}", seed, index, False))
                        rounds.append(round_)
                    result["timing_rounds"] = rounds
                    result.update(decide(quality, rounds, p))
            verify_files(campaign["frozen_files"])
            verify_files(campaign["baseline_runner_files"])
            verify_files(result["candidate_runner_files"])
            verify_files(result["source"]["files"])
            if model_snapshot(args.candidate, p, baseline) != candidate or model_snapshot(baseline["models"], p) != baseline:
                raise ValueError("Programs changed during the experiment")
            if environment() != campaign["environment"] or digest(campaign_path) != campaign_sha:
                raise ValueError("Environment or campaign changed during the experiment")
        except (Exception, KeyboardInterrupt) as error:
            result.update(status="crash", reason=f"{type(error).__name__}: {error}")
        result["completed_utc"] = datetime.now(timezone.utc).isoformat()
        write_json(output / "result.json", result)
        ledger = args.campaign / "results.tsv"
        fresh = not ledger.exists()
        with ledger.open("a", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream, delimiter="\t")
            if fresh:
                writer.writerow(["tag", "commit", "status", "gain", "baseline_drift", "description", "result"])
            writer.writerow([args.tag, result.get("source", {}).get("commit", ""), result["status"],
                             result.get("gain", ""), result.get("baseline_drift", ""),
                             args.description.replace("\t", " ").replace("\n", " ").replace("\r", " "), str((output / "result.json").resolve())])
        print(f"{result['status']}: {result['reason']}\n{output / 'result.json'}")
        return 1 if result["status"] == "crash" else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    init = sub.add_parser("init", help="Freeze evaluation inputs and preserve the baseline executable")
    init.add_argument("--campaign", type=Path, required=True)
    init.add_argument("--baseline", type=Path, default=ROOT / "artifacts/optimized")
    init.add_argument("--runner", type=Path, default=EXECUTABLE)
    init.add_argument("--protocol", type=Path, default=ROOT / "research/protocol.json")
    run = sub.add_parser("run", help="Qualify and benchmark one candidate without changing Git state")
    run.add_argument("--campaign", type=Path, required=True)
    run.add_argument("--candidate", type=Path, required=True)
    run.add_argument("--runner", type=Path, default=EXECUTABLE)
    run.add_argument("--tag", required=True)
    run.add_argument("--description", required=True)
    run.add_argument("--build-command", type=Path, help="JSON argv array; no shell parsing; separate build timeout")
    args = parser.parse_args()
    args.campaign = args.campaign.resolve()
    args.runner = args.runner.resolve()
    if args.command == "run":
        args.candidate = args.candidate.resolve()
    try:
        return initialize(args) if args.command == "init" else experiment(args)
    except (OSError, ValueError, KeyError) as error:
        parser.exit(1, f"{type(error).__name__}: {error}\n")


if __name__ == "__main__":
    raise SystemExit(main())
