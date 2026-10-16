"""Archive compact optimization evidence, including program hashes and failures."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np

from qnn_runtime import ROOT


def compact(path):
    report = json.loads(path.read_text())
    baseline = next(iter(report["variants"]))
    for case in report["cases"].values():
        base = json.loads(Path(case[baseline]["report"]).read_text())
        timing = base["runs"][-1]
        base_latent = np.fromfile(Path(case[baseline]["report"]).parent / "latent.raw", np.float32).reshape(
            1, 144, timing["bucket_length"])[..., :timing["latent_length"]]
        for entry in case.values():
            full_path = Path(entry["report"])
            full = json.loads(full_path.read_text())
            timing = full["runs"][-1]
            latent = np.fromfile(full_path.parent / "latent.raw", np.float32).reshape(
                1, 144, timing["bucket_length"])[..., :timing["latent_length"]]
            entry["flow_bit_exact_with_first_variant"] = bool(np.array_equal(latent, base_latent))
            entry["real_input_stages"] = full["real_input_stages"]
            entry["warm_p90_rtf"] = entry["warm"]["synthesis_ms_p90"] / (entry["duration_seconds"] * 1000)
            entry["report"] = full_path.relative_to(ROOT).as_posix()
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "benchmarks/optimization-2026-10-15.json")
    args = parser.parse_args()
    work = ROOT / "artifacts/optimization"
    result = {"archived_utc": datetime.now(timezone.utc).isoformat(),
              "device": "Snapdragon X2 Elite X2E84100 / SC8480XP / HTP V81",
              "qairt": "2.50.40.260831", "executorch": "1.5.0", "torch": "2.14.0", "torchao": "0.18.0",
              "python": "Global 3.14.7 x64, no venv; inference process native ARM64", "power_plan": "Balanced",
              "recommended": "artifacts/optimized (conservative FP16 preset), subject to the recorded absolute ONNX limitations and listening evaluation",
              "experiments": {
                  "bucketing": "64/96/128 latent frames; six initial cases had bit-exact audio versus the 128-frame baseline",
                  "layout": "Exact FP32 graph reformulation: keep ConvNeXt pointwise MLP in BLC layout; FP16 rounding changes",
                  "tanh_gelu": "Rejected as a full-pipeline speed improvement; 297.6 ms versus 296.0 ms at 96 frames in the screen",
                  "full_a16w8": "Rejected: vocoder held-out relative L2 5.215; calibration-case device error 6.419; eager PTQ calibration-sample error 0.124",
                  "affine_a16w8": "Rejected: 37 delegate partitions with CPU compute remaining; eager PTQ sample error 0.109",
                  "full_flow_layout": "Experimental: additional absolute numerical failures on short English and Portuguese at seed 42, Korean spectral error at seed 43"
              }, "measurements": {}, "programs": []}
    for name in ("final-8steps", "seed43-8steps", "final-5steps", "spanish-seed43-baseline"):
        result["measurements"][name] = compact(work / name / "summary.json")
    result["profiles"] = {kind: {name: json.loads((work / kind / name / "summary.json").read_text())
                                  for name in ("vector_estimator", "vocoder")}
                          for kind in ("profiles", "layout-profiles")}
    # Capture hashes before optional removal of rejected candidate programs.
    paths = list((ROOT / "artifacts/qnn").glob("*.pte")) + list(work.glob("*-l*/*.pte"))
    for path in sorted(paths):
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        result["programs"].append({"path": path.relative_to(ROOT).as_posix(), "sha256": digest,
                                   "metadata": json.loads(path.with_suffix(".json").read_text())})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
