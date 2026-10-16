"""Run matched speech/ONNX comparisons and warm timing for candidate bundles."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

from qnn_runtime import ROOT, metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", action="append", required=True, help="NAME=MODEL_DIRECTORY; first is baseline")
    parser.add_argument("--case", action="append")
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--repeat", type=int, default=10)
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/optimization/qualification")
    parser.add_argument("--timing-only", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--reverse-order", action="store_true", help="Reverse the alternating variant order for a matched repeat")
    args = parser.parse_args()
    if args.repeat < 3:
        parser.error("At least three requests are required")
    variants = dict(item.split("=", 1) for item in args.variant)
    baseline = next(iter(variants))
    cases = json.loads((ROOT / "scripts/optimization_cases.json").read_text(encoding="utf-8"))["qualification"]
    if args.case:
        cases = [case for case in cases if case["id"] in args.case]
        if len(cases) != len(set(args.case)):
            parser.error("Unknown qualification case")
    args.output.mkdir(parents=True, exist_ok=True)
    summary = {"started_utc": datetime.now(timezone.utc).isoformat(), "steps": args.steps, "seed": args.seed,
               "requests_per_case": args.repeat, "warm_requests_per_case": args.repeat - 1,
               "noise": "Identical seed at base capacity 128, cropped per channel by native runner",
               "timing": "Warm native synthesis; excludes loading, Python preprocessing and file I/O; profile collection disabled",
               "reverse_order": args.reverse_order,
               "limitations": "Small held-out corpus, one seed; numerical checks do not establish perceptual quality or WER",
               "numerical_gates": {"stage_relative_l2_max": 0.05, "final_latent_relative_l2_max": 0.05,
                                   "spectral_convergence_max": 0.10},
               "variants": variants, "cases": {}}
    for index, case in enumerate(cases):
        results = {}
        ordered = list(variants.items())
        if (index + args.reverse_order) % 2:
            ordered.reverse()
        for name, models in ordered:
            output = args.output / case["id"] / name
            output.mkdir(parents=True, exist_ok=True)
            report_path = output / "report.json"
            report_path.unlink(missing_ok=True)
            command = [sys.executable, str(ROOT / "scripts/synthesize.py"), "--models", models, "--output", str(output),
                       "--text", case["text"], "--voice", case["voice"], "--lang", case["lang"], "--steps", str(args.steps),
                       "--repeat", str(args.repeat), "--seed", str(args.seed)]
            if not args.timing_only:
                command.append("--compare")
            with (output / "run.log").open("w", encoding="utf-8") as log:
                completed = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
            report = json.loads(report_path.read_text()) if report_path.exists() else {}
            result = {"exit_code": completed.returncode, "report": str(report_path.resolve()),
                      "warm": report.get("warm_summary"), "duration_seconds": report.get("runs", [{}])[-1].get("duration_seconds"),
                      "bucket_length": report.get("runs", [{}])[-1].get("bucket_length"),
                      "spectral_convergence": report.get("spectral_convergence"),
                      "flow_relative_l2": report.get("flow_end_to_end", {}).get("relative_l2"),
                      "waveform_cosine": report.get("waveform_end_to_end", {}).get("cosine")}
            results[name] = result
            if not args.timing_only and report.get("flow_end_to_end"):
                result["numerical_gates_pass"] = (completed.returncode == 0 and result["flow_relative_l2"] <= 0.05
                                                  and result["spectral_convergence"] <= 0.10)
            print(case["id"], name, completed.returncode, result["warm"], flush=True)
        if results[baseline]["warm"]:
            base_ms = results[baseline]["warm"]["synthesis_ms_median"]
            base_report = json.loads(Path(results[baseline]["report"]).read_text())
            wav_length = base_report["runs"][-1]["wav_length"]
            base_audio = np.fromfile(Path(results[baseline]["report"]).parent / "audio.raw", np.float32)[:wav_length]
            base_timing = base_report["runs"][-1]
            base_latent = np.fromfile(Path(results[baseline]["report"]).parent / "latent.raw", np.float32).reshape(
                1, -1, base_timing["bucket_length"])[..., :base_timing["latent_length"]]
            for name, result in results.items():
                if result["warm"]:
                    result["latency_reduction_percent"] = 100 * (1 - result["warm"]["synthesis_ms_median"] / base_ms)
                    audio = np.fromfile(Path(result["report"]).parent / "audio.raw", np.float32)[:wav_length]
                    result["waveform_vs_baseline"] = metrics(audio, base_audio)
                    result["bit_exact_with_baseline"] = bool(np.array_equal(audio, base_audio))
                    candidate = json.loads(Path(result["report"]).read_text())
                    timing = candidate["runs"][-1]
                    latent = np.fromfile(Path(result["report"]).parent / "latent.raw", np.float32).reshape(
                        1, -1, timing["bucket_length"])[..., :timing["latent_length"]]
                    result["flow_vs_baseline"] = metrics(latent, base_latent)
                    result["flow_bit_exact_with_baseline"] = bool(np.array_equal(latent, base_latent))
                    result["flow_invocations"] = timing.get("flow_invocations", args.steps)
                    result["flow_unroll"] = timing.get("flow_unroll", 1)
        summary["cases"][case["id"]] = results
        (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    summary["completed_utc"] = datetime.now(timezone.utc).isoformat()
    summary["numerical_pass_by_variant"] = {
        name: all(case[name].get("numerical_gates_pass", False) for case in summary["cases"].values())
        for name in variants
    } if not args.timing_only else None
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(args.output / "summary.json", flush=True)
    if any(result["exit_code"] for case in summary["cases"].values() for result in case.values()):
        raise SystemExit(1)
    if summary["numerical_pass_by_variant"] and not all(summary["numerical_pass_by_variant"].values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
