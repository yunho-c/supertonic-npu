"""Compile and assemble the experimental T384/L128,256,512 Superfast preset.

Requires the pinned local assets, ExecuTorch checkout and QAIRT SDK. Completed
components can be resumed only after verifying their bytes, source and contract.
Incomplete components and published bundles are never overwritten.
"""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys

if __package__:
    from .export_superfast_context import ROOT, sha
else:
    from export_superfast_context import ROOT, sha

BUCKETS = (128, 256, 512)
COMPONENTS = [("duration_predictor", 128, "frontend"),
              ("text_encoder", 128, "frontend")]
COMPONENTS += [(name, length, f"l{length}") for length in reversed(BUCKETS)
               for name in ("vector_estimator", "vocoder")]


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def validate(directory, name, length, assets):
    record = read(directory / f"{name}.json")
    required = dict(component=name, precision="fp16", gelu="exact", profile_level="off",
                    text_length=384, latent_length=length, qnn_sdk="2.50.40.260831",
                    target_soc=88, htp_arch=81, delegates=1, cpu_compute=[])
    for key, expected in required.items():
        if record.get(key) != expected:
            raise ValueError(f"Invalid {directory}/{name}: {key}")
    path = directory / f"{name}.pte"
    if record.get("pte_sha256") != sha(path) or record.get("pte_bytes") != path.stat().st_size:
        raise ValueError(f"Program bytes changed: {path}")
    pinned = read(assets / "manifest.json")
    source = next(p["sha256"] for p in pinned["files"] if p["path"] == f"onnx/{name}.onnx")
    if record.get("model_sha256") != source:
        raise ValueError(f"Source weights changed: {name}")
    expected_sources = {str(p.relative_to(ROOT)): sha(p)
                        for p in sorted((ROOT / "scripts/superfast").glob("*.py"))}
    expected_sources["scripts/export_superfast_context.py"] = sha(ROOT / "scripts/export_superfast_context.py")
    if (record.get("export_sources_sha256") != expected_sources or
            record.get("source_transformations_sha256") != sha(ROOT / "scripts/superfast/sources.json")):
        raise ValueError(f"Export source differs from completed component: {name}")
    shapes = {
        "duration_predictor": [("text_ids", [1, 384], "int64"), ("style_dp", [1, 8, 16], "float32"),
                               ("text_mask", [1, 1, 384], "float32")],
        "text_encoder": [("text_ids", [1, 384], "int64"), ("style_ttl", [1, 50, 256], "float32"),
                         ("text_mask", [1, 1, 384], "float32")],
        "vector_estimator": [("noisy_latent", [1, 144, length], "float32"),
                             ("text_emb", [1, 256, 384], "float32"),
                             ("style_ttl", [1, 50, 256], "float32"),
                             ("latent_mask", [1, 1, length], "float32"),
                             ("text_mask", [1, 1, 384], "float32"),
                             ("current_step", [1], "float32"), ("total_step", [1], "float32")],
        "vocoder": [("latent", [1, 144, length], "float32")],
    }
    expected_inputs = [dict(name=n, shape=s, dtype=d) for n, s, d in shapes[name]]
    outputs = {"duration_predictor": [1], "text_encoder": [1, 256, 384],
               "vector_estimator": [1, 144, length], "vocoder": [1, length * 3072]}
    if record["inputs"] != expected_inputs or record["output"] != dict(shape=outputs[name], dtype="float32"):
        raise ValueError(f"Invalid tensor interface: {name}")
    if name in ("duration_predictor", "text_encoder"):
        valid = record.get("frontend_mode") == "pack8-norm-tail"
    elif name == "vector_estimator":
        valid = (record.get("flow_total_steps") == 8 and record.get("flow_unroll") == 1
                 and record.get("flow_output") == "latent"
                 and record.get("update_precision") == "htp-fp16"
                 and record.get("mlp_blocks") == list(range(28))
                 and record.get("fusion_blocks") == list(range(24))
                 and record.get("share_guidance_prefix") and record.get("gather_boundary")
                 and not record.get("collapse_unconditional_text")
                 and not record.get("reuse_conditioning") and not record.get("specialize_time"))
    else:
        valid = (record.get("conv_layout") == "channel-last" and
                 record.get("selection") == ["direct-prelu", "fold-input-affine", "fold-gamma"])
    if not valid:
        raise ValueError(f"Superfast transformations changed: {name}")
    return record


def build(output, assets, sdk, assemble_only=False):
    output = output.resolve()
    if (output / "bundle.json").exists() or (output / "provenance.json").exists():
        raise FileExistsError("Published preset exists; choose a fresh output directory")
    records = {}
    for name, length, folder in COMPONENTS:
        directory = output / folder
        metadata = directory / f"{name}.json"
        program = directory / f"{name}.pte"
        if not metadata.exists() and not program.exists() and not assemble_only:
            directory.mkdir(parents=True, exist_ok=True)
            with (directory / f"{name}_build.log").open("x", encoding="utf-8") as log:
                subprocess.run([sys.executable, str(ROOT / "scripts/export_superfast_context.py"),
                                "--component", name, "--text-length", "384", "--latent-length", str(length),
                                "--assets", str(assets.resolve()), "--qnn-sdk", str(sdk.resolve()),
                                "--output", str(directory)], stdout=log, stderr=subprocess.STDOUT, check=True)
        record = validate(directory, name, length, assets)
        records[f"{folder}/{name}"] = {
            "pte_sha256": record["pte_sha256"], "metadata_sha256": sha(metadata),
            "pte_bytes": record["pte_bytes"], "reference_metrics": record["reference_metrics"],
        }
        print(f"Verified {folder}/{name}", flush=True)
    # The existing native runner/client infer noise capacity from the base VE.
    # Use the real L512 base; retain independent frontend copies for provenance.
    for name in ("duration_predictor", "text_encoder"):
        for extension in ("json", "pte"):
            source = output / "frontend" / f"{name}.{extension}"
            target = output / "l512" / source.name
            if target.exists():
                if sha(source) != sha(target):
                    raise ValueError(f"Existing base frontend differs: {target}")
            else:
                shutil.copyfile(source, target)
    bundle = {"preset": "superfast-t384", "base": "l512", "noise_length": 512,
              "flow_total_steps": 8, "frontend_mode": "pack8-norm-tail", "frontend_text_capacity": 384,
              "buckets": [{"latent_length": length, "models": f"l{length}",
                           "flow_unroll": 1, "flow_total_steps": 8} for length in BUCKETS]}
    provenance = {"preset": "superfast-t384", "components": records,
                  "scope": "Experimental FP16, exact GELU, T384/L128,256,512, eight separate updates; listening pending"}
    for name, value in (("provenance.json", provenance), ("bundle.json", bundle)):
        with (output / name).open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(json.dumps(value, indent=2) + "\n")
    return bundle


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--assets", type=Path, default=ROOT / "assets")
    parser.add_argument("--qnn-sdk", type=Path, default=Path("C:/Qualcomm/AIStack/QAIRT/2.50.40.260831"))
    parser.add_argument("--assemble-only", action="store_true")
    args = parser.parse_args()
    print(json.dumps(build(args.output, args.assets, args.qnn_sdk, args.assemble_only), indent=2))
