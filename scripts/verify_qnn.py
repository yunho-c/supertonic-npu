"""Execute exported components on the physical HTP and compare ONNX outputs."""

import argparse
import json
from pathlib import Path

import numpy as np

from qnn_runtime import ROOT, NativeRunner, metrics, tensor_file


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", type=Path, default=ROOT / "artifacts/qnn")
    parser.add_argument("--component", action="append")
    args = parser.parse_args()
    output = args.models / "verification"
    report = {}
    with NativeRunner(args.models, output / "htp.log") as runner:
        for name in args.component or ["duration_predictor", "text_encoder", "vector_estimator", "vocoder"]:
            metadata = json.loads((args.models / f"{name}.json").read_text())
            reference = np.load(args.models / f"{name}_reference.npz")
            specs = [tensor_file(output, name + "_" + item["name"], reference[item["name"]]) for item in metadata["inputs"]]
            result_file = output / f"{name}_output.raw"
            timing = runner.request({"command": "infer", "component": name, "inputs": specs, "output": str(result_file), "repeat": 4})
            actual = np.fromfile(result_file, dtype=np.float32).reshape(reference["expected"].shape)
            comparison = metrics(actual, reference["expected"])
            report[name] = {"metrics": comparison, "timings": timing}
            print(name, json.dumps(report[name]), flush=True)
            (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
            if comparison["relative_l2"] > 0.05:
                raise AssertionError(f"{name}: HTP relative L2 error exceeds 5%")


if __name__ == "__main__":
    main()
