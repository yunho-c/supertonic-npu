"""Export and validate grouped FP16 flow steps; does not replace existing programs."""

import argparse
import collections
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
import onnxruntime as ort
import torch

from examples.models.supertonic.export import common
from examples.models.supertonic.loaders.checkpoint_loader import load_vector_estimator
from examples.models.supertonic.model.config import TTSConfig
from examples.models.supertonic.source_transformations.mlx import exportable_vector_estimator, replace_same_padding
from flow_unroll import UnrolledFlow
from qnn_runtime import DEFAULT_SDK, metrics
from qnn_transformations import optimize_modules, prepare_vector_graph, replace_masked_padding


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--assets", type=Path, default=ROOT / "assets")
    parser.add_argument("--qnn-sdk", type=Path, default=DEFAULT_SDK)
    parser.add_argument("--unroll", type=int, choices=(1, 2, 4, 8), default=2)
    parser.add_argument("--schedule-steps", type=int, default=8)
    parser.add_argument("--reuse-conditioning", action="store_true")
    parser.add_argument("--specialize-time", action="store_true")
    parser.add_argument("--conv-layout", choices=("original", "pointwise-2d", "gelu-last"), default="original")
    parser.add_argument("--text-length", type=int, default=128)
    parser.add_argument("--latent-length", type=int, default=96)
    parser.add_argument("--reference-only", action="store_true")
    parser.add_argument("--optrace", action="store_true", help="Build a separate profiling program and HTP schematic")
    args = parser.parse_args()
    args.output = args.output.resolve()
    args.qnn_sdk = args.qnn_sdk.resolve()
    if args.schedule_steps != 8:
        parser.error("This prototype qualifies the eight-step schedule only")
    if args.specialize_time and not args.reuse_conditioning:
        parser.error("--specialize-time requires --reuse-conditioning")
    if (args.output / "vector_estimator.pte").exists():
        parser.error("Output already contains a compiled program; choose another directory")
    torch.set_num_threads(4)
    torch.manual_seed(42)
    args.output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    config = TTSConfig.from_json(args.assets / "onnx/tts.json")
    estimator = load_vector_estimator(args.assets / "onnx/vector_estimator.onnx", config).eval()
    estimator = exportable_vector_estimator(replace_masked_padding(replace_same_padding(estimator)))
    estimator = optimize_modules(estimator, conv_layout=args.conv_layout)
    model = UnrolledFlow(estimator, args.unroll, args.schedule_steps, args.reuse_conditioning, args.specialize_time).eval()
    inputs = common.example_inputs(config, common.ExportBounds(args.text_length, args.latent_length))["vector_estimator"]
    names = common.method_contracts(config)["vector_estimator"].input_names
    inputs = tuple(x.float() if x.is_floating_point() else x for x in inputs)
    named = dict(zip(names, inputs))
    valid_t, valid_l = max(1, args.text_length - 17), max(1, args.latent_length - 23)
    named["text_mask"][..., valid_t:] = 0
    named["latent_mask"][..., valid_l:] = 0
    named["current_step"].fill_(0)
    named["total_step"].fill_(args.schedule_steps)
    feed = {key: value.numpy() for key, value in named.items()}
    reference = {key: (value[..., :valid_t] if key in ("text_emb", "text_mask") else
                       value[..., :valid_l] if key in ("noisy_latent", "latent_mask") else value)
                 for key, value in feed.items()}
    options = ort.SessionOptions(); options.intra_op_num_threads = 4
    session = ort.InferenceSession(str(args.assets / "onnx/vector_estimator.onnx"), sess_options=options,
                                   providers=["CPUExecutionProvider"])
    with torch.no_grad():
        actual = model(*inputs).numpy()
    for offset in range(args.unroll):
        reference["current_step"] = np.full_like(reference["current_step"], offset)
        reference["noisy_latent"] = session.run(None, reference)[0]
    expected = np.pad(reference["noisy_latent"], ((0, 0), (0, 0), (0, args.latent_length - valid_l)))
    comparison = metrics(actual, expected)
    if comparison["relative_l2"] > 2e-5 or comparison["max_error"] > 2e-3:
        raise AssertionError(f"Grouped FP32 graph differs from original ONNX: {comparison}")
    print(f"FP32 vs {args.unroll} original ONNX steps: {comparison}", flush=True)
    np.savez(args.output / "vector_estimator_reference.npz", **feed, expected=expected)
    manifest = json.loads((args.assets / "manifest.json").read_text())
    record = {"component": "vector_estimator", "precision": "fp16", "gelu": "exact", "conv_layout": args.conv_layout,
              "profile_level": "optrace" if args.optrace else "off",
              "flow_unroll": args.unroll, "flow_total_steps": args.schedule_steps,
              "reuse_conditioning": args.reuse_conditioning, "specialize_time": args.specialize_time,
              "qnn_sdk": args.qnn_sdk.name, "target_soc": 88, "htp_arch": 81,
              "text_length": args.text_length, "latent_length": args.latent_length,
              "model_sha256": next(x["sha256"] for x in manifest["files"] if x["path"] == "onnx/vector_estimator.onnx"),
              "reference": "Repeated original unpadded ONNX with eight-step schedule, starting at zero",
              "reference_metrics": comparison, "valid_text_length": valid_t, "valid_latent_length": valid_l,
              "inputs": [{"name": k, "shape": list(v.shape), "dtype": str(v.dtype)} for k, v in feed.items()],
              "output": {"shape": list(expected.shape), "dtype": str(expected.dtype)},
              "unique_parameter_bytes": sum(p.numel() * p.element_size() for p in model.parameters()),
              "time_constant_bytes": sum(b.numel() * b.element_size() for k, b in model.named_buffers() if k.startswith("time_condition_")),
              "export_sources_sha256": {name: hashlib.sha256((ROOT / "scripts" / name).read_bytes()).hexdigest()
                                         for name in ("export_flow.py", "flow_unroll.py", "qnn_transformations.py")}}
    del session
    # Retain the validated reference/configuration even if a later compiler
    # failure prevents creation of a usable program. Reuse requires a .pte and
    # successful delegation metadata, not just this preliminary record.
    (args.output / "vector_estimator.json").write_text(json.dumps(record, indent=2) + "\n")
    if not args.reference_only:
        os.environ["QNN_SDK_ROOT"] = str(args.qnn_sdk)
        os.environ["FLATC_EXECUTABLE"] = str(ROOT / "build/executorch-host/third-party/flatc_ep/bin/flatc.exe")
        sdk_lib = args.qnn_sdk / "lib/x86_64-windows-msvc"
        os.environ["PATH"] = str(sdk_lib) + os.pathsep + os.environ["PATH"]
        with os.add_dll_directory(str(sdk_lib)):
            from executorch.backends.qualcomm.serialization.qc_schema import QcomChipset
            from executorch.backends.qualcomm.utils.utils import generate_htp_compiler_spec, generate_qnn_executorch_compiler_spec, to_edge_transform_and_lower_to_qnn
            from executorch.exir import ExecutorchBackendConfig
            print("Capturing grouped graph", flush=True)
            model = prepare_vector_graph(model, inputs)
            record["captured_operations"] = dict(collections.Counter(str(n.target) for n in model.graph.nodes if n.op == "call_function"))
            specs = generate_qnn_executorch_compiler_spec(soc_model=QcomChipset.SC8480XP,
                        backend_options=generate_htp_compiler_spec(use_fp16=True), profile_level=3 if args.optrace else 0)
            compile_start = time.perf_counter()
            print("Compiling QNN graph", flush=True)
            # QNN writes its Optrace schematic in the compiler's working directory.
            with contextlib.chdir(args.output):
                edge = to_edge_transform_and_lower_to_qnn(model, inputs, specs)
            if args.optrace:
                schematics = sorted(args.output.glob("*_schematic.bin"))
                if len(schematics) != 1:
                    raise RuntimeError(f"Expected one HTP schematic, found {schematics}")
                record["schematic"] = schematics[0].name
            graph = edge.exported_program().graph
            (args.output / "vector_estimator_graph.txt").write_text(str(graph), encoding="utf-8")
            compute = [str(n.target) for n in graph.nodes if n.op == "call_function"]
            fallback = [op for op in compute if "executorch_call_delegate" not in op and "getitem" not in op]
            delegates = sum("executorch_call_delegate" in op for op in compute)
            if fallback or delegates != 1:
                raise RuntimeError(f"Grouped flow requires one QNN delegate: {delegates=}, {fallback=}")
            program = edge.to_executorch(config=ExecutorchBackendConfig(extract_delegate_segments=True))
            path = args.output / "vector_estimator.pte"
            with path.open("wb") as stream:
                program.write_to_file(stream)
            record.update(delegates=delegates, cpu_compute=fallback, compile_seconds=time.perf_counter() - compile_start,
                          pte_bytes=path.stat().st_size)
    record["total_seconds"] = time.perf_counter() - started
    (args.output / "vector_estimator.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps({k: v for k, v in record.items() if k != "captured_operations"}, indent=2), flush=True)


if __name__ == "__main__":
    main()
