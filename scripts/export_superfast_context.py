"""Export one FP16 Superfast component with explicit text/audio capacities.

Keeps exact GELU, pack8-norm-tail frontend, full28-mlp-only flow with shared CFG
prefix and eight separate updates, and the measured combined vocoder. The FP32
export is checked against original ONNX before any HTP compilation. Numerical
device differences and subjective listening are qualified separately.
"""
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


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def export(args):
    import numpy as np
    import onnxruntime as ort
    import torch
    from examples.models.supertonic.export import common
    from examples.models.supertonic.loaders import checkpoint_loader
    from examples.models.supertonic.model.config import TTSConfig
    from examples.models.supertonic.source_transformations.mlx import (
        exportable_vector_estimator, replace_relative_attention,
        replace_same_padding, replace_vocoder_causal_padding,
    )
    from superfast.qnn_transformations import (
        optimize_modules, prepare_vector_graph, replace_masked_padding,
    )
    from qnn_runtime import metrics

    name = args.component
    args.output = args.output.resolve()
    if (args.output / f"{name}.json").exists() or (args.output / f"{name}.pte").exists():
        raise FileExistsError("Choose fresh component output; existing evidence is immutable")
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.manual_seed(42)
    started = time.perf_counter()
    config = TTSConfig.from_json(args.assets / "onnx/tts.json")
    model_path = args.assets / "onnx" / f"{name}.onnx"
    source_hash = sha(model_path)
    pinned = json.loads((args.assets / "manifest.json").read_text())
    expected_hash = next(p["sha256"] for p in pinned["files"] if p["path"] == f"onnx/{name}.onnx")
    if source_hash != expected_hash:
        raise ValueError("Source ONNX hash does not match the asset manifest")
    print(f"Loading {name}: T{args.text_length}/L{args.latent_length}", flush=True)
    model = getattr(checkpoint_loader, "load_" + name)(model_path, config).eval()
    model = replace_masked_padding(replace_same_padding(model))
    transformation = {}
    if name in ("duration_predictor", "text_encoder"):
        from superfast.frontend_transformations import apply_frontend
        model = apply_frontend(replace_relative_attention(model), "pack8-norm-tail", transformation)
        transformation["frontend_text_capacity"] = args.text_length
    elif name == "vector_estimator":
        from superfast.research_transformations import apply_experiment
        from superfast.fusion_transformations import apply_fusion
        from superfast.final_four_transformations import extend_final_four
        from superfast.flow_unroll import UnrolledFlow
        model = optimize_modules(exportable_vector_estimator(model), conv_layout="gelu-last")
        model = apply_experiment(model, "mlp-pack8", mlp_exclude=[24, 25, 26, 27],
                                 gather_boundary=True, metadata=transformation)
        model = apply_fusion(model, "packed-norm-tail", metadata=transformation)
        model = extend_final_four(model, norm_tail=False, metadata=transformation)
        model = UnrolledFlow(model, unroll=1, schedule_steps=8, share_guidance_prefix=True).eval()
        transformation.update(
            final_four_mode="full28-mlp-only", conv_layout="gelu-last",
            flow_unroll=1, flow_total_steps=8, flow_output="latent", update_precision="htp-fp16",
            share_guidance_prefix=True, collapse_unconditional_text=False,
            reuse_conditioning=False, specialize_time=False,
        )
    else:
        from superfast.vocoder_transformations import transform
        selection = ["direct-prelu", "fold-input-affine", "fold-gamma"]
        model = optimize_modules(replace_vocoder_causal_padding(model), conv_layout="channel-last")
        model = transform(model, "combined", selection).eval()
        transformation.update(mode="combined", selection=selection, conv_layout="channel-last")

    bounds = common.ExportBounds(args.text_length, args.latent_length)
    inputs = tuple(x.float() if x.is_floating_point() else x for x in common.example_inputs(config, bounds)[name])
    names = common.method_contracts(config)[name].input_names
    valid_text, valid_latent = args.text_length - 17, args.latent_length - 23
    for key, value in zip(names, inputs):
        if key == "text_mask":
            value[..., valid_text:] = 0
        elif key == "latent_mask":
            value[..., valid_latent:] = 0
        elif key == "current_step":
            value.fill_(0)
        elif key == "total_step":
            value.fill_(8)
    feed = {key: value.numpy() for key, value in zip(names, inputs)}
    reference = {}
    for key, value in feed.items():
        if key in ("text_ids", "text_emb", "text_mask"):
            value = value[..., :valid_text]
        elif key in ("noisy_latent", "latent_mask"):
            value = value[..., :valid_latent]
        reference[key] = value
    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    session = ort.InferenceSession(str(model_path), sess_options=options, providers=["CPUExecutionProvider"])
    expected = session.run(None, reference)[0]
    if name == "text_encoder":
        expected = np.pad(expected, ((0, 0), (0, 0), (0, args.text_length - valid_text)))
    elif name == "vector_estimator":
        expected = np.pad(expected, ((0, 0), (0, 0), (0, args.latent_length - valid_latent)))
    with torch.inference_mode():
        actual = model(*inputs).numpy()
    comparison = metrics(actual, expected)
    if not np.isfinite(actual).all() or comparison["relative_l2"] > 2e-5 or comparison["max_error"] > 2e-3:
        raise AssertionError(f"FP32 transformed graph differs from original ONNX: {comparison}")
    print(f"FP32 source parity: {comparison}", flush=True)
    np.savez(args.output / f"{name}_reference.npz", **feed, expected=expected)
    del session, actual
    sources = {str(p.relative_to(ROOT)): sha(p) for p in sorted((ROOT / "scripts/superfast").glob("*.py"))}
    sources["scripts/export_superfast_context.py"] = sha(__file__)
    record = {
        "component": name, "precision": "fp16", "gelu": "exact", "profile_level": "off",
        "text_length": args.text_length, "latent_length": args.latent_length,
        "qnn_sdk": args.qnn_sdk.name, "target_soc": 88, "htp_arch": 81,
        "model_sha256": source_hash, "source_transformations": "scripts/superfast/sources.json",
        "source_transformations_sha256": sha(ROOT / "scripts/superfast/sources.json"),
        "export_sources_sha256": sources,
        "reference": "original FP32 ONNX; unpadded text/VE, full fixed vocoder input",
        "valid_text_length": valid_text, "valid_latent_length": valid_latent,
        "reference_metrics": comparison,
        "inputs": [{"name": key, "shape": list(value.shape), "dtype": str(value.dtype)} for key, value in feed.items()],
        "output": {"shape": list(expected.shape), "dtype": str(expected.dtype)},
        **transformation,
    }
    metadata = args.output / f"{name}.json"
    metadata.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    if not args.reference_only:
        os.environ["QNN_SDK_ROOT"] = str(args.qnn_sdk)
        os.environ["FLATC_EXECUTABLE"] = str(ROOT / "build/executorch-host/third-party/flatc_ep/bin/flatc.exe")
        sdk_lib = args.qnn_sdk / "lib/x86_64-windows-msvc"
        os.environ["PATH"] = str(sdk_lib) + os.pathsep + os.environ["PATH"]
        with os.add_dll_directory(str(sdk_lib)):
            from executorch.backends.qualcomm.serialization.qc_schema import QcomChipset
            from executorch.backends.qualcomm.utils.utils import (
                generate_htp_compiler_spec, generate_qnn_executorch_compiler_spec,
                to_edge_transform_and_lower_to_qnn,
            )
            from executorch.exir import ExecutorchBackendConfig
            if name == "vector_estimator":
                model = prepare_vector_graph(model, inputs)
            specs = generate_qnn_executorch_compiler_spec(
                soc_model=QcomChipset.SC8480XP,
                backend_options=generate_htp_compiler_spec(use_fp16=True), profile_level=0,
            )
            compile_start = time.perf_counter()
            print("Compiling QNN HTP graph", flush=True)
            with contextlib.chdir(args.output):
                edge = to_edge_transform_and_lower_to_qnn(model, inputs, specs)
            graph = edge.exported_program().graph
            (args.output / f"{name}_graph.txt").write_text(str(graph), encoding="utf-8")
            compute = [str(n.target) for n in graph.nodes if n.op == "call_function"]
            fallback = [op for op in compute if "executorch_call_delegate" not in op and "getitem" not in op]
            delegates = sum("executorch_call_delegate" in op for op in compute)
            if delegates != 1 or fallback:
                raise RuntimeError(f"Expected one complete HTP delegate: {delegates=}, {fallback=}")
            program = edge.to_executorch(config=ExecutorchBackendConfig(extract_delegate_segments=True))
            path = args.output / f"{name}.pte"
            with path.open("wb") as stream:
                program.write_to_file(stream)
            record.update(delegates=delegates, cpu_compute=fallback,
                          compile_seconds=time.perf_counter() - compile_start,
                          pte_bytes=path.stat().st_size, pte_sha256=sha(path),
                          delegated_operations=dict(collections.Counter(compute)))
    record["total_seconds"] = time.perf_counter() - started
    metadata.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(f"Finished {name} in {record['total_seconds']:.1f}s", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--component", required=True, choices=("duration_predictor", "text_encoder", "vector_estimator", "vocoder"))
    parser.add_argument("--text-length", type=int, choices=(128, 384), default=384)
    parser.add_argument("--latent-length", type=int, choices=(96, 128, 256, 512), default=128)
    parser.add_argument("--assets", type=Path, default=ROOT / "assets")
    parser.add_argument("--qnn-sdk", type=Path, default=Path("C:/Qualcomm/AIStack/QAIRT/2.50.40.260831"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference-only", action="store_true")
    export(parser.parse_args())


if __name__ == "__main__":
    main()
