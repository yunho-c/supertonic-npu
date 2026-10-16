"""Retain grouped-flow measurements, program hashes, and compilation evidence."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from qnn_runtime import ROOT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", type=Path, default=ROOT / "artifacts/flow")
    parser.add_argument("--output", type=Path, default=ROOT / "benchmarks/flow-unroll-2026-10-15.json")
    args = parser.parse_args()
    args.work = args.work.resolve()
    result = {"archived_utc": datetime.now(timezone.utc).isoformat(),
              "device": "Snapdragon X2 Elite X2E84100 / SC8480XP / HTP V81",
              "precision": "FP16 HTP, FP32 public tensors; original convolution layout",
              "schedule_steps": 8, "qairt": "2.50.40.260831", "executorch": "1.5.0",
              "python": "Global 3.14.7 x64 without a virtualenv; native ARM64 inference",
              "status": "Experimental; the existing conservative preset is unchanged",
              "limitations": "Numerical screening is not perceptual qualification. Host traces do not measure HTP activity. No compiler or profiler runs during timing screens.",
              "programs": [], "measurements": {}, "profiles": {}, "host_traces": {}}
    for path in sorted(args.work.glob("*-l*/vector_estimator.json")):
        entry = {"path": path.relative_to(ROOT).as_posix(), "metadata": json.loads(path.read_text())}
        binary = path.with_suffix(".pte")
        entry["compiled"] = binary.exists()
        if binary.exists():
            with binary.open("rb") as stream:
                entry["pte_sha256"] = hashlib.file_digest(stream, "sha256").hexdigest()
        result["programs"].append(entry)
    for path in sorted(args.work.glob("*/summary.json")):
        report = json.loads(path.read_text())
        if "cases" not in report or "variants" not in report:
            continue
        if path.parent.name.startswith("quality-"):
            report["purpose"] = "Secondary-seed numerical checks; too few warm requests for performance conclusions"
        for variants in report["cases"].values():
            for entry in variants.values():
                source = Path(entry["report"])
                if source.exists():
                    runs = json.loads(source.read_text()).get("runs", [])
                    entry["warm_samples_ms"] = [
                        {"synthesis": run["inference_ms"], "flow": sum(run["vector_estimator_ms"])}
                        for run in runs[1:]]
                entry["report"] = source.relative_to(ROOT).as_posix()
        result["measurements"][path.parent.name] = report
    for path in sorted(args.work.glob("profiles/*/vector_estimator/summary.json")):
        result["profiles"][path.parents[1].name] = {
            "summary": json.loads(path.read_text()),
            "provenance": json.loads(path.with_name("provenance.json").read_text())}
    for path in sorted(args.work.glob("trace-*/host_trace.json")):
        events = json.loads(path.read_text())["traceEvents"]
        report = json.loads(path.with_name("report.json").read_text())
        result["host_traces"][path.parent.name] = {
            "path": path.relative_to(ROOT).as_posix(), "events": len(events),
            "flow_calls_per_request": [run["flow_invocations"] for run in report["runs"]],
            "warning": "Host wall time including synchronous backend calls; not HTP hardware activity"}
    guards = args.work / "runtime-guards/report.json"
    if guards.exists():
        result["rejected_invalid_schedule_inputs"] = json.loads(guards.read_text())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
