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
    counts = collections.Counter()
    nonzero = collections.Counter()
    for key, value in means.items():
        operation = re.sub(r"_\d+$", "", key.split(":")[0])
        groups[operation] += value
        counts[operation] += 1
        nonzero[operation] += value > 0
    total = sum(means.values())
    return {"warning": "Detailed profiling changes execution time; percentages are summed node cycles, not production latency",
            "total_node_cycles": total,
            "top_groups": [{"operation": key, "nodes": counts[key], "nonzero_nodes": nonzero[key], "cycles": value,
                            "percent_node_cycles": 100 * value / total} for key, value in groups.most_common(15)],
            "top_nodes": [{"node": key, "cycles": value, "percent_node_cycles": 100 * value / total}
                          for key, value in sorted(means.items(), key=lambda item: -item[1])[:20]]}


def summarize_optrace(output):
    """Retain the physical GELU kernels and shapes without archiving the graph."""
    graph = json.loads((output / "chrometrace_htp.json").read_text())["graph"]
    gelu = []
    for node in graph["nodes"].values():
        qnn_types = node.get("scalar_params", {}).get("qnn_op_type", {}).values()
        if "Gelu" in qnn_types:
            gelu.append(node)
    shapes = collections.Counter()
    for node in gelu:
        if node["type"].startswith("q::Gelu"):
            for tensor in node["output_names"]:
                shapes[tuple(graph["tensors"][tensor]["dims"])] += 1
    runtrace = json.loads((output / "chrometrace_runtrace.json").read_text())["traceEvents"]
    files = {}
    for filename in ("chrometrace.json", "chrometrace_htp.json", "chrometrace_runtrace.json"):
        path = output / filename
        with path.open("rb") as stream:
            files[filename] = {"bytes": path.stat().st_size, "sha256": hashlib.file_digest(stream, "sha256").hexdigest()}
    return {"warning": "Physical compiled graph; kernel tiles are not logical operator counts. Instrumentation changes execution time. Trace coordinates use backend cycles/ticks; use Duration (cycles) and summary_time_us attributes.",
            "physical_node_types": dict(collections.Counter(n["type"] for n in graph["nodes"].values())),
            "gelu_source_ops": len({n["grouping"] for n in gelu}),
            "gelu_physical_node_types": dict(collections.Counter(n["type"] for n in gelu)),
            "gelu_kernel_output_shapes": [{"shape": list(shape), "tiles": count} for shape, count in shapes.items()],
            "graph_execute_us": [e["args"]["summary_time_us"] for e in runtrace if e.get("name") == "Graph Execute"],
            "files": files}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", type=Path, default=ROOT / "artifacts/qnn")
    parser.add_argument("--component", choices=["vector_estimator", "vocoder"], required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/optimization/profiles")
    parser.add_argument("--repeat", type=int, default=10)
    parser.add_argument("--keep-context", action="store_true", help="Retain the extracted context copy for direct QNN experiments")
    parser.add_argument("--optrace", action="store_true", help="Capture an HTP timeline; requires a program built with --optrace")
    args = parser.parse_args()
    args.models = args.models.resolve()
    schematic = None
    if args.optrace:
        schematics = sorted(args.models.glob("*_schematic.bin"))
        if len(schematics) != 1:
            parser.error("--optrace requires exactly one *_schematic.bin beside the matching program")
        schematic = schematics[0]
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
    if args.optrace:
        command.extend(["--profiling_option", "optrace"])
    with (output / "run.log").open("w") as log:
        subprocess.run(command, env=runtime_environment(), stdout=log, stderr=subprocess.STDOUT, check=True)
    with (output / "profile.txt").open("w") as log:
        subprocess.run([str(binaries / "qnn-profile-viewer.exe"), "--input_log", str(output / "qnn-profiling-data.log"),
                        "--output", str(output / "profile.csv")], env=runtime_environment(), stdout=log, stderr=subprocess.STDOUT, check=True)
    trace_command = None
    if args.optrace:
        config = output / "optrace-config.json"
        config.write_text(json.dumps({"features": {"htp_json": True, "runtrace": True,
                                                  "memory_info": True, "traceback": True, "qhas_json": True}}, indent=2) + "\n")
        trace_command = [str(binaries / "qnn-profile-viewer.exe"),
                         "--reader", str(DEFAULT_SDK / "lib/aarch64-windows-msvc/QnnHtpOptraceProfilingReader.dll"),
                         "--input_log", str(output / "qnn-profiling-data.log"), "--schematic", str(schematic),
                         "--output", str(output / "chrometrace.json"), "--config", str(config)]
        with (output / "optrace-viewer.log").open("w") as log:
            subprocess.run(trace_command, env=runtime_environment(), cwd=output,
                           stdout=log, stderr=subprocess.STDOUT, check=True)
    (output / "provenance.json").write_text(json.dumps({"models": str(args.models.resolve()), "metadata": metadata,
        "pte_sha256": hashlib.sha256((args.models / f"{args.component}.pte").read_bytes()).hexdigest(),
        "context_retained": args.keep_context,
        "repeat": args.repeat, "inputs": "Exporter synthetic component reference; profiling only, not an RTF benchmark",
        "command": command, "optrace_command": trace_command,
        "schematic_sha256": hashlib.sha256(schematic.read_bytes()).hexdigest() if schematic else None}, indent=2) + "\n")
    expected = reference["expected"]
    actual = np.fromfile(next((output / "Result_0").glob("*.raw")), np.float32).reshape(expected.shape)
    summary = summarize_profile(output / "profile.csv")
    summary["output_vs_onnx"] = metrics(actual, expected)
    if args.optrace:
        summary["htp_trace"] = "chrometrace.json"
        summary["trace_warning"] = "Instrumented device HTP execution on synthetic component input, not synthesis latency"
        summary["optrace"] = summarize_optrace(output)
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    if not args.keep_context:
        context.unlink()
    print(output / "profile.csv")


if __name__ == "__main__":
    main()
