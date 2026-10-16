"""Build experimental grouped-flow programs and reference existing text/vocoder models."""

import argparse
import json
from pathlib import Path
import subprocess
import sys

from make_bundle import make_bundle
from qnn_runtime import ROOT

VARIANTS = {
    "plain2": (2, False, False), "plain4": (4, False, False), "plain8": (8, False, False),
    "reuse2": (2, True, False), "reuse4": (4, True, False), "reuse8": (8, True, False),
    "special2": (2, True, True), "special4": (4, True, True), "special8": (8, True, True),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=VARIANTS, action="append", required=True)
    parser.add_argument("--bucket", type=int, choices=(64, 96, 128), action="append")
    parser.add_argument("--work", type=Path, default=ROOT / "artifacts/flow")
    parser.add_argument("--reuse", action="store_true")
    args = parser.parse_args()
    lengths = sorted(set(args.bucket or [96]))
    base = ROOT / "artifacts/qnn"
    expected_hash = json.loads((base / "vector_estimator.json").read_text())["model_sha256"]
    vocoders = [ROOT / f"artifacts/optimization/layout-l{length}" for length in lengths]
    for directory in vocoders:
        if not all((directory / f"vocoder.{suffix}").is_file() for suffix in ("json", "pte")):
            raise FileNotFoundError(f"Missing layout vocoder in {directory}; run scripts/build_optimized.py --reuse first")
    args.work.mkdir(parents=True, exist_ok=True)
    for variant in args.variant:
        unroll, reuse, specialize = VARIANTS[variant]
        directories = []
        for length in lengths:
            directory = args.work / f"{variant}-l{length}"
            metadata = directory / "vector_estimator.json"
            if args.reuse and metadata.exists() and (directory / "vector_estimator.pte").exists():
                record = json.loads(metadata.read_text())
                expected = {"flow_unroll": unroll, "flow_total_steps": 8, "reuse_conditioning": reuse,
                            "specialize_time": specialize, "latent_length": length, "text_length": 128,
                            "precision": "fp16", "conv_layout": "original", "gelu": "exact",
                            "delegates": 1, "cpu_compute": [], "model_sha256": expected_hash}
                if any(record.get(k) != v for k, v in expected.items()):
                    raise ValueError(f"Existing program does not match requested flow variant: {directory}")
                print(f"Reusing {directory}", flush=True)
            else:
                command = [sys.executable, str(ROOT / "scripts/export_flow.py"), "--unroll", str(unroll),
                           "--latent-length", str(length), "--output", str(directory)]
                if reuse:
                    command.append("--reuse-conditioning")
                if specialize:
                    command.append("--specialize-time")
                log_path = args.work / f"{variant}-l{length}-build.log"
                print(f"Building {variant} L{length}; log: {log_path}", flush=True)
                with log_path.open("w") as log:
                    subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
            directories.append(directory)
        # Suffix makes single-bucket screens distinct from complete presets.
        preset = args.work / (variant + "-buckets" + "-".join(map(str, lengths)))
        make_bundle(base, directories, preset, vocoders)


if __name__ == "__main__":
    main()
