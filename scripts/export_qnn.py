"""Export fixed-shape Supertonic 3 components for the X2 Elite HTP."""

import argparse
import gc
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
ET_SOURCE = ROOT / ".deps" / "executorch"
sys.path[:0] = [str(ET_SOURCE.parent), str(ET_SOURCE)]

import numpy as np
import onnxruntime as ort
import torch

from examples.models.supertonic.export import common
from examples.models.supertonic.loaders import checkpoint_loader
from examples.models.supertonic.model.config import TTSConfig
from examples.models.supertonic.source_transformations.mlx import (
    exportable_vector_estimator,
    replace_relative_attention,
    replace_same_padding,
    replace_vocoder_causal_padding,
)
from qnn_transformations import prepare_vector_graph, replace_masked_padding


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets", type=Path, default=ROOT / "assets")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts" / "qnn")
    parser.add_argument("--qnn-sdk", type=Path, default=Path(r"C:\Qualcomm\AIStack\QAIRT\2.50.40.260831"))
    parser.add_argument("--component", choices=common.METHOD_NAMES, action="append")
    parser.add_argument("--text-length", type=int, default=128)
    parser.add_argument("--latent-length", type=int, default=128)
    parser.add_argument("--reference-only", action="store_true")
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(4)
    args.output.mkdir(parents=True, exist_ok=True)
    config = TTSConfig.from_json(args.assets / "onnx/tts.json")
    bounds = common.ExportBounds(args.text_length, args.latent_length)
    samples = common.example_inputs(config, bounds)
    contracts = common.method_contracts(config)
    asset_manifest = json.loads((args.assets / "manifest.json").read_text())
    asset_hashes = {item["path"]: item["sha256"] for item in asset_manifest["files"]}
    os.environ["QNN_SDK_ROOT"] = str(args.qnn_sdk)
    os.environ["FLATC_EXECUTABLE"] = str(ROOT / "build/executorch-host/third-party/flatc_ep/bin/flatc.exe")
    sdk_lib = args.qnn_sdk / "lib/x86_64-windows-msvc"
    os.environ["PATH"] = str(sdk_lib) + os.pathsep + os.environ["PATH"]
    dll_handle = os.add_dll_directory(str(sdk_lib))
    for name in args.component or common.METHOD_NAMES:
        print(f"Loading {name}", flush=True)
        model_path = args.assets / "onnx" / f"{name}.onnx"
        model = getattr(checkpoint_loader, f"load_{name}")(model_path, config).eval()
        model = replace_same_padding(model)
        model = replace_masked_padding(model)
        if name in ("duration_predictor", "text_encoder"):
            model = replace_relative_attention(model)
        elif name == "vector_estimator":
            model = exportable_vector_estimator(model)
        elif name == "vocoder":
            model = replace_vocoder_causal_padding(model)
        inputs = tuple(value.float() if value.is_floating_point() else value for value in samples[name])
        input_names = contracts[name].input_names
        valid_text = max(1, args.text_length - 17)
        valid_latent = max(1, args.latent_length - 23)
        for key, value in zip(input_names, inputs):
            if key == "text_mask":
                value[..., valid_text:] = 0
            elif key == "latent_mask":
                value[..., valid_latent:] = 0
        if name == "vector_estimator":
            model = prepare_vector_graph(model, inputs)
        named_inputs = {key: value.numpy() for key, value in zip(input_names, inputs)}
        # Compare fixed buckets against the original unpadded model, since its
        # replicate-padding semantics depend on the physical sequence boundary.
        reference_inputs = {}
        for key, value in named_inputs.items():
            if key in ("text_ids", "text_mask", "text_emb"):
                value = value[..., :valid_text]
            elif key in ("noisy_latent", "latent_mask"):
                value = value[..., :valid_latent]
            reference_inputs[key] = value
        options = ort.SessionOptions()
        options.intra_op_num_threads = 4
        session = ort.InferenceSession(str(model_path), sess_options=options, providers=["CPUExecutionProvider"])
        expected = session.run(None, reference_inputs)[0]
        if name == "text_encoder":
            expected = np.pad(expected, ((0, 0), (0, 0), (0, args.text_length - valid_text)))
        elif name == "vector_estimator":
            expected = np.pad(expected, ((0, 0), (0, 0), (0, args.latent_length - valid_latent)))
        with torch.no_grad():
            actual = model(*inputs).numpy()
        np.testing.assert_allclose(actual, expected, atol=2e-4, rtol=2e-4)
        error = actual.astype(np.float64) - expected.astype(np.float64)
        record = {
            "component": name,
            "model_sha256": asset_hashes[f"onnx/{name}.onnx"],
            "qnn_sdk": args.qnn_sdk.name,
            "target_soc": 88,
            "htp_arch": 81,
            "precision": "fp16",
            "text_length": args.text_length,
            "latent_length": args.latent_length,
            "reference": "original ONNX with unpadded valid sequences",
            "valid_text_length": valid_text,
            "valid_latent_length": valid_latent,
            "reference_max_error": float(np.abs(error).max()),
            "reference_rmse": float(np.sqrt(np.mean(error * error))),
            "inputs": [{"name": key, "shape": list(value.shape), "dtype": str(value.dtype)} for key, value in named_inputs.items()],
            "output": {"shape": list(expected.shape), "dtype": str(expected.dtype)},
        }
        print(f"{name} PyTorch/ONNX parity: {record['reference_max_error']:.8g} max error", flush=True)
        np.savez(args.output / f"{name}_reference.npz", **named_inputs, expected=expected)
        del session
        if not args.reference_only:
            from executorch.backends.qualcomm.serialization.qc_schema import QcomChipset
            from executorch.backends.qualcomm.utils.utils import (
                generate_htp_compiler_spec,
                generate_qnn_executorch_compiler_spec,
                to_edge_transform_and_lower_to_qnn,
            )
            from executorch.exir import ExecutorchBackendConfig

            specs = generate_qnn_executorch_compiler_spec(
                soc_model=QcomChipset.SC8480XP,
                backend_options=generate_htp_compiler_spec(use_fp16=True),
                profile_level=2 if args.profile else 0,
            )
            start = time.perf_counter()
            edge = to_edge_transform_and_lower_to_qnn(model, inputs, specs)
            graph = edge.exported_program().graph
            (args.output / f"{name}_graph.txt").write_text(str(graph), encoding="utf-8")
            compute = [str(node.target) for node in graph.nodes if node.op == "call_function"]
            fallback = [op for op in compute if "executorch_call_delegate" not in op and "getitem" not in op]
            delegates = sum("executorch_call_delegate" in op for op in compute)
            if fallback or delegates == 0:
                raise RuntimeError(f"{name}: {delegates} delegates, CPU compute remains: {fallback}")
            program = edge.to_executorch(config=ExecutorchBackendConfig(extract_delegate_segments=True))
            output_path = args.output / f"{name}.pte"
            with output_path.open("wb") as output:
                program.write_to_file(output)
            record.update({"delegates": delegates, "cpu_compute": fallback, "compile_seconds": time.perf_counter() - start, "pte_bytes": output_path.stat().st_size})
            print(f"Saved {output_path}: {delegates} QNN delegate(s), {record['compile_seconds']:.1f} s", flush=True)
            del edge, program
        (args.output / f"{name}.json").write_text(json.dumps(record, indent=2) + "\n")
        del model
        gc.collect()
    summary = [json.loads(path.read_text()) for name in common.METHOD_NAMES
               if (path := args.output / f"{name}.json").exists()]
    (args.output / "export_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    dll_handle.close()


if __name__ == "__main__":
    main()
