"""Build shared FP16 bucket programs and conservative/fast synthesis presets."""

import argparse
import json
from pathlib import Path
import subprocess
import sys

from make_bundle import make_bundle
from qnn_runtime import ROOT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=ROOT / "artifacts/qnn")
    parser.add_argument("--work", type=Path, default=ROOT / "artifacts/optimization")
    parser.add_argument("--reuse", action="store_true", help="Reuse existing programs whose metadata matches this configuration")
    args = parser.parse_args()
    if sys.version_info[:2] != (3, 14) or sys.prefix != sys.base_prefix:
        raise RuntimeError("Use global Python 3.14, without a virtual environment")
    base_metadata = json.loads((args.base / "vector_estimator.json").read_text())
    if base_metadata["text_length"] != 128 or base_metadata["latent_length"] != 128:
        raise ValueError("This preset builder expects the default 128-position base")
    args.work.mkdir(parents=True, exist_ok=True)
    original, layout = [], []
    for length in (64, 96, 128):
        for kind, conv_layout in (("fp16", "original"), ("layout", "channel-last")):
            directory = args.base if kind == "fp16" and length == 128 else args.work / f"{kind}-l{length}"
            for component in ("vector_estimator", "vocoder"):
                record_path = directory / f"{component}.json"
                if directory == args.base:
                    continue
                if args.reuse and record_path.exists() and (directory / f"{component}.pte").exists():
                    record = json.loads(record_path.read_text())
                    source = json.loads((args.base / f"{component}.json").read_text())
                    if (record["latent_length"] == length and record["text_length"] == 128
                            and record["precision"] == "fp16" and record.get("gelu", "exact") == "exact"
                            and record.get("conv_layout", "original") == conv_layout
                            and record["model_sha256"] == source["model_sha256"]
                            and record["qnn_sdk"] == source["qnn_sdk"] and not record["cpu_compute"]):
                        print(f"Reusing {directory.name}/{component}", flush=True)
                        continue
                log_path = args.work / f"build-{kind}-{length}-{component}.log"
                print(f"Exporting {kind} {length} {component}; log: {log_path}", flush=True)
                with log_path.open("w", encoding="utf-8") as log:
                    subprocess.run([sys.executable, str(ROOT / "scripts/export_qnn.py"), "--component", component,
                                    "--latent-length", str(length), "--conv-layout", conv_layout, "--output", str(directory)],
                                   stdout=log, stderr=subprocess.STDOUT, check=True)
            (original if kind == "fp16" else layout).append(directory)
    make_bundle(args.base, original, ROOT / "artifacts/optimized", vocoders=layout)
    make_bundle(args.base, layout, ROOT / "artifacts/optimized-fast")


if __name__ == "__main__":
    main()
