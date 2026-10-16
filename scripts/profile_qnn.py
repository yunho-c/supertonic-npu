"""Collect native QNN per-operator profiles from an existing ExecuTorch program."""

import argparse
import collections
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import re

import numpy as np

from qnn_runtime import DEFAULT_SDK, ROOT, metrics, runtime_environment, tensor_file


def summarize_profile(path):
    nodes = collections.defaultdict(list)
    with path.open() as stream:
        for _ in range(4):
            next(stream)
        for row in csv.DictReader(stream, skipinitialspace=True):
            if row["Message"] == "EXECUTE" and row["Event Level"] == "SUB-EVENT" and row["Unit of Measurement"] == "CYCLES":
                nodes[row["Event Identifier"]].append(float(row["Time"]))
    means = {key: float(np.mean(values[1:] or values)) for key, values in nodes.items()}
    groups = collections.Counter()
    for key, value in means.items():
        groups[re.sub(r"_\d+$", "", key.split(":")[0])] += value
    total = sum(means.values())
    return {"warning": "Detailed profiling changes execution time; percentages are summed node cycles, not production latency",
            "top_groups": [{"operation": key, "percent_node_cycles": 100 * value / total} for key, value in groups.most_common(15)],
            "top_nodes": [{"node": key, "cycles": value, "percent_node_cycles": 100 * value / total}
                          for key, value in sorted(means.items(), key=lambda item: -item[1])[:20]]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", type=Path, default=ROOT / "artifacts/qnn")
    parser.add_argument("--component", choices=["vector_estimator", "vocoder"], required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/optimization/profiles")
    parser.add_argument("--repeat", type=int, default=10)
    parser.add_argument("--keep-context", action="store_true", help="Retain the extracted context copy for direct QNN experiments")
    args = parser.parse_args()
    sys.path[:0] = [str(ROOT / ".deps"), str(ROOT / ".deps/executorch")]
    os.environ["QNN_SDK_ROOT"] = str(DEFAULT_SDK)
    os.environ["FLATC_EXECUTABLE"] = str(ROOT / "build/executorch-host/third-party/flatc_ep/bin/flatc.exe")
    with os.add_dll_directory(str(DEFAULT_SDK / "lib/x86_64-windows-msvc")):
        from executorch.exir._serialize._program import deserialize_pte_binary
        from executorch.backends.qualcomm.python import PyQnnManagerAdaptor
        program = deserialize_pte_binary((args.models / f"{args.component}.pte").read_bytes()).program
        delegates = program.execution_plan[0].delegates
        if len(delegates) != 1 or delegates[0].id != "QnnBackend":
            raise ValueError("Profiling requires exactly one QNN delegate")
        data = program.backend_delegate_data[delegates[0].processed.index].data
        binary = PyQnnManagerAdaptor.StripProtocol(data)
    output = args.output.resolve() / args.component
    output.mkdir(parents=True, exist_ok=True)
    context = output / "context.bin"
    context.write_bytes(binary)
    metadata = json.loads((args.models / f"{args.component}.json").read_text())
    reference = np.load(args.models / f"{args.component}_reference.npz")
    paths = [tensor_file(output / "inputs", entry["name"], reference[entry["name"]])["path"]
             for entry in metadata["inputs"]]
    input_list = output / "inputs.txt"
    input_list.write_text(" ".join(f"input_{index}_{entry['name']}_0:={path}" for index, (entry, path) in enumerate(zip(metadata["inputs"], paths))) + "\n")
    binaries = DEFAULT_SDK / "bin/aarch64-windows-msvc"
    command = [str(binaries / "qnn-net-run.exe"), "--backend", str(DEFAULT_SDK / "lib/aarch64-windows-msvc/QnnHtp.dll"),
               "--retrieve_context", str(context), "--input_list", str(input_list), "--output_dir", str(output),
               "--profiling_level", "detailed", "--perf_profile", "burst", "--num_inferences", str(args.repeat),
               "--keep_num_outputs", "1"]
    with (output / "run.log").open("w") as log:
        subprocess.run(command, env=runtime_environment(), stdout=log, stderr=subprocess.STDOUT, check=True)
    with (output / "profile.txt").open("w") as log:
        subprocess.run([str(binaries / "qnn-profile-viewer.exe"), "--input_log", str(output / "qnn-profiling-data.log"),
                        "--output", str(output / "profile.csv")], env=runtime_environment(), stdout=log, stderr=subprocess.STDOUT, check=True)
    (output / "provenance.json").write_text(json.dumps({"models": str(args.models.resolve()), "metadata": metadata,
        "pte_sha256": hashlib.sha256((args.models / f"{args.component}.pte").read_bytes()).hexdigest(),
        "context_retained": args.keep_context,
        "repeat": args.repeat, "inputs": "Exporter synthetic component reference; profiling only, not an RTF benchmark",
        "command": command}, indent=2) + "\n")
    expected = reference["expected"]
    actual = np.fromfile(next((output / "Result_0").glob("*.raw")), np.float32).reshape(expected.shape)
    summary = summarize_profile(output / "profile.csv")
    summary["output_vs_onnx"] = metrics(actual, expected)
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    if not args.keep_context:
        context.unlink()
    print(output / "profile.csv")


if __name__ == "__main__":
    main()
