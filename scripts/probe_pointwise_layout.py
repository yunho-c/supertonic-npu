"""Isolated HTP experiment for the first flow block's learned pointwise MLP.

Compiles separate diagnostic programs, then measures forward and reverse orders.
This is not an end-to-end speech benchmark or a qualified model transformation.
"""

import argparse
import contextlib
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / ".deps"), str(ROOT / ".deps/executorch")]

import numpy as np
import torch
from torch.nn import functional as F

from examples.models.supertonic.loaders.checkpoint_loader import load_vector_estimator
from examples.models.supertonic.model.config import TTSConfig
from qnn_runtime import DEFAULT_SDK, NativeRunner, metrics, tensor_file


class Pointwise(torch.nn.Module):
    def __init__(self, block, mode):
        super().__init__()
        self.mode = mode
        self.first = block.pwconv1
        self.second = block.pwconv2
        self.act = block.act
        if mode in ("pointwise2d", "packed8"):
            self.w1 = torch.nn.Parameter(block.pwconv1.weight.detach().unsqueeze(-1))
            self.w2 = torch.nn.Parameter(block.pwconv2.weight.detach().unsqueeze(-1))

    def forward(self, x):
        if self.mode == "original":
            return self.second(self.act(self.first(x)))
        if self.mode == "gelu_last":
            y = self.first(x)
            return self.second(self.act(y.transpose(1, 2)).transpose(1, 2))
        if self.mode == "linear":
            y = F.linear(x.transpose(1, 2), self.first.weight[..., 0], self.first.bias)
            return F.linear(self.act(y), self.second.weight[..., 0], self.second.bias).transpose(1, 2)
        if self.mode == "pointwise2d":
            y = F.conv2d(x.unsqueeze(2), self.w1, self.first.bias)
            return F.conv2d(self.act(y), self.w2, self.second.bias).squeeze(2)
        batch, channels, length = x.shape
        y = x.transpose(1, 2).reshape(1, 8, batch * length // 8, channels).permute(0, 3, 1, 2)
        y = F.conv2d(y, self.w1, self.first.bias)
        y = F.conv2d(self.act(y), self.w2, self.second.bias)
        return y.permute(0, 2, 3, 1).reshape(batch, length, channels).transpose(1, 2)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/first-principles/pointwise-probe")
    args = parser.parse_args()
    args.output = args.output.resolve()
    if args.output.exists():
        parser.error("Choose a fresh output directory to preserve existing evidence")
    args.output.mkdir(parents=True)
    torch.set_num_threads(2)
    torch.manual_seed(20261016)
    config = TTSConfig.from_json(ROOT / "assets/onnx/tts.json")
    estimator = load_vector_estimator(ROOT / "assets/onnx/vector_estimator.onnx", config).eval()
    block = estimator.vector_field.main_blocks[0].convnext[0]
    sdk_lib = DEFAULT_SDK / "lib/x86_64-windows-msvc"
    os.environ["QNN_SDK_ROOT"] = str(DEFAULT_SDK)
    os.environ["FLATC_EXECUTABLE"] = str(ROOT / "build/executorch-host/third-party/flatc_ep/bin/flatc.exe")
    os.environ["PATH"] = str(sdk_lib) + os.pathsep + os.environ["PATH"]
    modes = ("original", "gelu_last", "pointwise2d", "packed8", "linear")
    programs = []
    with os.add_dll_directory(str(sdk_lib)), torch.inference_mode():
        from executorch.backends.qualcomm.serialization.qc_schema import QcomChipset
        from executorch.backends.qualcomm.utils.utils import generate_htp_compiler_spec, generate_qnn_executorch_compiler_spec, to_edge_transform_and_lower_to_qnn
        from executorch.exir import ExecutorchBackendConfig
        specs = generate_qnn_executorch_compiler_spec(
            soc_model=QcomChipset.SC8480XP,
            backend_options=generate_htp_compiler_spec(use_fp16=True), profile_level=0)
        for length in (64, 96, 128):
            x = torch.randn(2, 512, length)
            expected = Pointwise(block, "original")(x).numpy()
            for mode in modes:
                directory = args.output / f"{mode}-l{length}"
                directory.mkdir()
                model = Pointwise(block, mode).eval()
                eager = metrics(model(x).numpy(), expected)
                if eager["relative_l2"] > 2e-5:
                    raise AssertionError(eager)
                start = time.perf_counter()
                with contextlib.chdir(directory):
                    edge = to_edge_transform_and_lower_to_qnn(model, (x,), specs)
                graph = edge.exported_program().graph
                compute = [str(node.target) for node in graph.nodes if node.op == "call_function"]
                fallback = [op for op in compute if "executorch_call_delegate" not in op and "getitem" not in op]
                delegates = sum("executorch_call_delegate" in op for op in compute)
                if delegates != 1 or fallback:
                    raise RuntimeError((delegates, fallback))
                program = edge.to_executorch(config=ExecutorchBackendConfig(extract_delegate_segments=True))
                path = directory / "vector_estimator.pte"
                with path.open("wb") as stream:
                    program.write_to_file(stream)
                metadata = {"component": "vector_estimator", "diagnostic_only": True,
                            "description": "First ConvNeXt pointwise MLP only, not a vector estimator",
                            "mode": mode, "length": length, "precision": "fp16", "profile_level": "off",
                            "delegates": delegates, "cpu_compute": fallback,
                            "compile_seconds": time.perf_counter() - start,
                            "eager_vs_original": eager, "pte_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                (directory / "vector_estimator.json").write_text(json.dumps(metadata, indent=2))
                np.save(directory / "expected.npy", expected)
                input_spec = tensor_file(directory, "input", x.numpy())
                programs.append({"directory": directory, "input": input_spec, "metadata": metadata})
                print(f"COMPILED {mode} {length}", flush=True)
    # All compilation finishes before any timing is collected.
    rows = []
    for order, sequence in (("forward", programs), ("reverse", list(reversed(programs)))):
        for item in sequence:
            directory = item["directory"]
            with NativeRunner(directory, directory / f"{order}.log") as runner:
                timing = runner.request({"command": "infer", "component": "vector_estimator",
                                         "inputs": [item["input"]], "repeat": 45,
                                         "output": str(directory / f"{order}.raw")})
            expected = np.load(directory / "expected.npy")
            actual = np.fromfile(directory / f"{order}.raw", dtype=np.float32).reshape(expected.shape)
            values = timing["vector_estimator_ms"][5:]
            row = {"mode": item["metadata"]["mode"], "length": item["metadata"]["length"],
                   "order": order, "warm_count": len(values), "median_ms": float(np.median(values)),
                   "p90_ms": float(np.percentile(values, 90)), "warm_ms": values,
                   "output_vs_fp32": metrics(actual, expected), "metadata": item["metadata"]}
            rows.append(row)
            print("MEASURED", json.dumps({k: row[k] for k in ("mode", "length", "order", "median_ms")}), flush=True)
    result = {"device": "Snapdragon X2 Elite X2E84100 / HTP V81", "qairt": DEFAULT_SDK.name,
              "qualification": "Synthetic isolated first-block MLP, FP16, profile off, native ARM64 forward wall time including launch and boundary handling. Forty warm measurements per order after five discarded calls. Separate loaded process per variant. Not an end-to-end speech benchmark.",
              "rows": rows}
    (args.output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(args.output / "summary.json", flush=True)


if __name__ == "__main__":
    main()
