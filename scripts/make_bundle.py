"""Reference existing programs in an automatically selected bucket bundle."""

import argparse
import json
import os
from pathlib import Path


def make_bundle(base, buckets, output, vocoders=()):
    base, output = base.resolve(), output.resolve()
    text = json.loads((base / "text_encoder.json").read_text())
    maximum = json.loads((base / "vector_estimator.json").read_text())["latent_length"]
    entries = {}
    schedules = set()
    overrides = {json.loads((directory / "vocoder.json").read_text())["latent_length"]: directory.resolve()
                 for directory in vocoders}
    for directory in buckets:
        directory = directory.resolve()
        vector = json.loads((directory / "vector_estimator.json").read_text())
        unroll, schedule = vector.get("flow_unroll", 1), vector.get("flow_total_steps", 0)
        if (type(unroll) is not int or type(schedule) is not int or unroll not in (1, 2, 4, 8)
                or not 0 <= schedule <= 100 or (unroll > 1 and not schedule) or (schedule and schedule % unroll)):
            raise ValueError(f"Invalid compiled flow schedule in {directory}")
        if schedule:
            schedules.add(schedule)
        length = vector["latent_length"]
        vocoder_dir = overrides.get(length, directory)
        vocoder = json.loads((vocoder_dir / "vocoder.json").read_text())
        if length != vocoder["latent_length"] or vector["text_length"] != text["text_length"]:
            raise ValueError(f"Incompatible shapes in {directory}")
        if length > maximum or length in entries:
            raise ValueError(f"Duplicate bucket or length exceeds base noise capacity: {length}")
        for name, record, model_dir in (("vector_estimator", vector, directory), ("vocoder", vocoder, vocoder_dir)):
            if record.get("profile_level", "off") != "off":
                raise ValueError(f"Profiling programs cannot be used in synthesis bundles: {model_dir}")
            original = json.loads((base / f"{name}.json").read_text())
            if record["model_sha256"] != original["model_sha256"] or record["cpu_compute"]:
                raise ValueError(f"Incompatible weights or CPU fallback in {directory}")
            if not (model_dir / f"{name}.pte").is_file():
                raise FileNotFoundError(model_dir / f"{name}.pte")
        entries[length] = {"latent_length": length, "models": os.path.relpath(directory, output)}
        if vocoder_dir != directory:
            entries[length]["vocoder"] = os.path.relpath(vocoder_dir, output)
        if schedule:
            entries[length].update(flow_unroll=unroll, flow_total_steps=schedule)
    if not entries:
        raise ValueError("At least one bucket is required")
    if set(overrides) - set(entries):
        raise ValueError("A vocoder override has no matching vector-estimator bucket")
    if len(schedules) > 1:
        raise ValueError("Bucket programs require different fixed flow schedules")
    bundle = {"base": os.path.relpath(base, output), "noise_length": maximum,
              "buckets": [entries[length] for length in sorted(entries)]}
    if schedules:
        bundle["flow_total_steps"] = schedules.pop()
    output.mkdir(parents=True, exist_ok=True)
    (output / "bundle.json").write_text(json.dumps(bundle, indent=2) + "\n")
    print(json.dumps(bundle, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=Path("artifacts/qnn"))
    parser.add_argument("--bucket", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--vocoder", type=Path, action="append", default=[], help="Override vocoder for its matching latent bucket")
    args = parser.parse_args()
    make_bundle(args.base, args.bucket, args.output, args.vocoder)
