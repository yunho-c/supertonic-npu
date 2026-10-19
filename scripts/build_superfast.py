"""Assemble the measured eight-step Superfast components without recompiling.

References immutable research programs in place. Requires the existing frontend,
full28-mlp-only and combined-vocoder artifacts. Only the L96/T128 configuration
has been combined; longer audio requests are rejected by the native runner.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESEARCH = ROOT / "artifacts/research"
FRONTEND = RESEARCH / "worktree-frontend-layout-oct16/artifacts/frontend-layout-oct16/campaign/bundles/pack8-norm-tail"
FLOW = RESEARCH / "worktree-final-four-oct16/artifacts/final-four-oct16/campaign/bundles/full28-mlp-only"
VOCODER = RESEARCH / "worktree-vocoder-oct16/artifacts/vocoder-head-oct16/bundles/combined"


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def bucket(directory):
    data = read(directory / "bundle.json")
    if data.get("flow_total_steps") != 8 or data.get("noise_length") != 128:
        raise ValueError("Superfast requires eight steps and capacity128 noise")
    entries = [b for b in data["buckets"] if b["latent_length"] == 96]
    if len(entries) != 1 or entries[0].get("flow_unroll") != 1 or entries[0].get("flow_total_steps") != 8:
        raise ValueError("Expected one L96 bucket with eight separate evaluations")
    return data, entries[0]


def build(frontend=FRONTEND, flow=FLOW, vocoder=VOCODER, output=ROOT / "artifacts/superfast", reference=ROOT / "artifacts/qnn"):
    frontend, flow, vocoder, output, reference = map(Path, (frontend, flow, vocoder, output, reference))
    if (output / "bundle.json").exists() or (output / "provenance.json").exists():
        raise FileExistsError(f"Preset already exists: {output}; choose a fresh output directory")
    front_bundle, _ = bucket(frontend)
    _, flow_bucket = bucket(flow)
    _, vocoder_bucket = bucket(vocoder)
    base = (frontend / front_bundle["base"]).resolve()
    flow_dir = (flow / flow_bucket["models"]).resolve()
    vocoder_dir = (vocoder / vocoder_bucket.get("vocoder", vocoder_bucket["models"])).resolve()
    directories = {"duration_predictor": base, "text_encoder": base,
                   "vector_estimator": flow_dir, "vocoder": vocoder_dir}
    metadata, identities = {}, {}
    for name, directory in directories.items():
        path = directory / f"{name}.pte"
        record = read(path.with_suffix(".json"))
        original = read(reference / f"{name}.json")
        if (record.get("component") != name or record.get("precision") != "fp16"
                or record.get("gelu", "exact") != "exact" or record.get("text_length") != 128
                or record.get("profile_level", "off") != "off"
                or record.get("delegates") != 1 or record.get("cpu_compute") != []):
            raise ValueError(f"Invalid production FP16/T128 delegation: {name}")
        for key in ("model_sha256", "qnn_sdk", "target_soc", "htp_arch"):
            if record.get(key) is None or record[key] != original.get(key):
                raise ValueError(f"Incompatible {key}: {name}")
        digest = sha(path)
        if record.get("pte_sha256", digest) != digest:
            raise ValueError(f"Compiled program hash mismatch: {name}")
        identities[name] = {"path": str(path), "pte_sha256": digest,
                            "metadata_sha256": sha(path.with_suffix(".json"))}
        metadata[name] = record
    for name in ("duration_predictor", "text_encoder"):
        if metadata[name].get("frontend_mode") != "pack8-norm-tail":
            raise ValueError(f"Expected packed frontend winner: {name}")
    f = metadata["vector_estimator"]
    if (f.get("latent_length") != 96 or f.get("flow_total_steps") != 8 or f.get("flow_unroll") != 1
            or f.get("mlp_blocks") != list(range(28)) or f.get("fusion_blocks") != list(range(24))
            or not f.get("share_guidance_prefix") or not f.get("gather_boundary")
            or f.get("collapse_unconditional_text") or f.get("reuse_conditioning") or f.get("specialize_time")):
        raise ValueError("Expected full28-mlp-only flow with shared CFG prefix and packed first24 norm/tail")
    v = metadata["vocoder"]
    if (v.get("latent_length") != 96 or v.get("conv_layout") != "channel-last"
            or set(v.get("selection", [])) != {"direct-prelu", "fold-input-affine", "fold-gamma"}):
        raise ValueError("Expected combined channel-last vocoder winner")
    base_meta = read(base / "vector_estimator.json")
    if (base_meta["text_length"], base_meta["latent_length"]) != (128, 128):
        raise ValueError("Frontend base must retain T128 and original capacity128 noise")
    # Check producer/consumer tensor layouts, beyond the nominal capacities.
    def inputs(name):
        return {x["name"]: (x["shape"], x["dtype"]) for x in metadata[name]["inputs"]}
    text_output = metadata["text_encoder"]["output"]
    flow_output = f["output"]
    if (text_output["shape"], text_output["dtype"]) != inputs("vector_estimator")["text_emb"]:
        raise ValueError("Text encoder output is incompatible with flow")
    if (flow_output["shape"], flow_output["dtype"]) != inputs("vocoder")["latent"]:
        raise ValueError("Flow output is incompatible with vocoder")
    for name in ("text_ids", "text_mask"):
        if inputs("duration_predictor")[name] != inputs("text_encoder")[name]:
            raise ValueError(f"Frontend inputs disagree: {name}")
    bundle = {"preset": "superfast", "base": os.path.relpath(base, output), "noise_length": 128,
              "flow_total_steps": 8, "frontend_mode": "pack8-norm-tail", "frontend_text_capacity": 128,
              "buckets": [{"latent_length": 96, "models": os.path.relpath(flow_dir, output),
                           "vocoder": os.path.relpath(vocoder_dir, output), "flow_unroll": 1, "flow_total_steps": 8}]}
    provenance = {"preset": "superfast", "scope": "FP16, exact GELU, T128/L96, eight evaluations; opt-in experimental preset",
                  "components": identities, "base_metadata_sha256": sha(base / "vector_estimator.json"),
                  "source_bundles": {str((p / "bundle.json").resolve()): sha(p / "bundle.json") for p in (frontend, flow, vocoder)}}
    output.mkdir(parents=True, exist_ok=True)
    for name, value in (("bundle.json", bundle), ("provenance.json", provenance)):
        (output / name).write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    return bundle


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name, default in (("frontend", FRONTEND), ("flow", FLOW), ("vocoder", VOCODER),
                          ("output", ROOT / "artifacts/superfast"), ("reference", ROOT / "artifacts/qnn")):
        parser.add_argument("--" + name, type=Path, default=default)
    args = parser.parse_args()
    print(json.dumps(build(**vars(args)), indent=2))


if __name__ == "__main__":
    main()
