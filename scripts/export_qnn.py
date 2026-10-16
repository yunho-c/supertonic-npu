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
from qnn_transformations import optimize_modules, prepare_vector_graph, replace_masked_padding


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
    parser.add_argument("--precision", choices=["fp16", "a16w8"], default="fp16")
    parser.add_argument("--calibration", type=Path)
    parser.add_argument("--gelu", choices=["exact", "tanh"], default="exact")
    parser.add_argument("--conv-layout", choices=["original", "channel-last"], default="original")
    parser.add_argument("--quant-scope", choices=["all", "affine"], default="all")
    parser.add_argument("--ptq-only", action="store_true", help="Calibrate and check eager PTQ without compiling a program")
    args = parser.parse_args()
    if args.precision != "fp16" and args.calibration is None:
        parser.error("Quantization requires representative --calibration data")
    if args.ptq_only and (args.precision == "fp16" or args.reference_only):
        parser.error("--ptq-only requires quantization and cannot be combined with --reference-only")
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
        model = optimize_modules(model, args.gelu, args.conv_layout)
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
        if args.gelu == "exact":
            np.testing.assert_allclose(actual, expected, atol=2e-4, rtol=2e-4)
        else:
            relative_l2 = np.linalg.norm(actual - expected) / max(np.linalg.norm(expected), 1e-30)
            if not np.isfinite(actual).all() or relative_l2 > 0.01:
                raise AssertionError(f"GELU approximation exceeds 1% reference relative L2: {relative_l2}")
        error = actual.astype(np.float64) - expected.astype(np.float64)
        record = {
            "component": name,
            "model_sha256": asset_hashes[f"onnx/{name}.onnx"],
            "qnn_sdk": args.qnn_sdk.name,
            "target_soc": 88,
            "htp_arch": 81,
            "precision": "mixed-a16w8-fp16" if args.precision == "a16w8" and args.quant_scope == "affine" else args.precision,
            "gelu": args.gelu,
            "conv_layout": args.conv_layout,
            "text_length": args.text_length,
            "latent_length": args.latent_length,
            "reference": "original ONNX with unpadded valid sequences",
            "valid_text_length": valid_text,
            "valid_latent_length": valid_latent,
            "reference_max_error": float(np.abs(error).max()),
            "reference_rmse": float(np.sqrt(np.mean(error * error))),
            "reference_relative_l2": float(np.linalg.norm(error) / max(np.linalg.norm(expected.astype(np.float64)), 1e-30)),
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

            if args.precision == "a16w8":
                from executorch.backends.qualcomm.quantizer.quantizer import QnnQuantizer, QuantDtype
                from torchao.quantization.pt2e.quantize_pt2e import prepare_pt2e, convert_pt2e
                from torchao.quantization.pt2e.observer import MinMaxObserver
                prefix = {"vector_estimator": "vector", "vocoder": "vocoder"}.get(name)
                if prefix is None:
                    raise ValueError("PTQ prototype currently covers vector estimator and vocoder only")
                calibration_manifest = json.loads((args.calibration / "manifest.json").read_text(encoding="utf-8"))
                if not calibration_manifest.get("complete", True):
                    raise ValueError("Calibration collection did not complete")
                calibration_hashes = {item["path"]: item["sha256"] for item in calibration_manifest["asset_manifest"]["files"]}
                if any(calibration_hashes.get(path) != value for path, value in asset_hashes.items()):
                    raise ValueError("Calibration assets differ from export assets")
                if calibration_manifest["latent_length"] != args.latent_length or calibration_manifest["text_length"] != args.text_length:
                    raise ValueError("Calibration shapes do not match the export")
                calibration_paths = ([args.calibration / filename for filename in calibration_manifest["samples"][name]]
                                     if "samples" in calibration_manifest else sorted(args.calibration.glob(f"{prefix}_*.npz")))
                if not calibration_paths:
                    raise ValueError("No calibration samples")
                quantizer = QnnQuantizer(soc_model=QcomChipset.SC8480XP)
                quantizer.set_default_quant_config(QuantDtype.use_16a8w, is_conv_per_channel=True,
                                                  is_linear_per_channel=True, act_observer=MinMaxObserver)
                if args.quant_scope == "affine":
                    affine = {torch.ops.aten.linear.default, torch.ops.aten.conv1d.default,
                              torch.ops.aten.conv2d.default, torch.ops.aten.convolution.default}
                    quantizer.add_discard_ops(list(quantizer.quant_ops - affine))
                prepared = prepare_pt2e(torch.export.export(model, inputs, strict=True).module(), quantizer)
                with torch.no_grad():
                    for index, path in enumerate(calibration_paths):
                        with np.load(path) as sample:
                            prepared(*(torch.from_numpy(sample[key]) for key in input_names))
                        if index % 16 == 0:
                            print(f"Calibration {index + 1}/{len(calibration_paths)}", flush=True)
                model = convert_pt2e(prepared)
                record["calibration"] = {"manifest": calibration_manifest, "samples": len(calibration_paths),
                                         "observer": "MinMaxObserver", "weights": "per-channel 8-bit", "activations": "16-bit",
                                         "scope": args.quant_scope}
                with np.load(calibration_paths[0]) as sample, torch.no_grad():
                    calibration_inputs = tuple(torch.from_numpy(sample[key]) for key in input_names)
                    original_session = ort.InferenceSession(str(model_path), sess_options=options, providers=["CPUExecutionProvider"])
                    original_feed = {key: sample[key] for key in input_names}
                    if name == "vector_estimator":
                        valid_l = int(sample["latent_mask"].sum())
                        valid_t = int(sample["text_mask"].sum())
                        for key in ("noisy_latent", "latent_mask"):
                            original_feed[key] = original_feed[key][..., :valid_l]
                        for key in ("text_emb", "text_mask"):
                            original_feed[key] = original_feed[key][..., :valid_t]
                    unquantized = original_session.run(None, original_feed)[0]
                    if name == "vector_estimator":
                        unquantized = np.pad(unquantized, ((0, 0), (0, 0), (0, args.latent_length - valid_l)))
                    quantized = model(*calibration_inputs).numpy()
                    record["ptq_calibration_sample_relative_l2"] = float(np.linalg.norm(quantized - unquantized) / np.linalg.norm(unquantized))
                    print(f"PTQ eager sample relative L2: {record['ptq_calibration_sample_relative_l2']:.6g}", flush=True)
                    del original_session
                if args.ptq_only:
                    (args.output / f"{name}.json").write_text(json.dumps(record, indent=2) + "\n")
                    del model, prepared
                    gc.collect()
                    continue
            specs = generate_qnn_executorch_compiler_spec(
                soc_model=QcomChipset.SC8480XP,
                backend_options=generate_htp_compiler_spec(use_fp16=args.precision == "fp16" or args.quant_scope == "affine"),
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
