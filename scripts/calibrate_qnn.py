"""Save representative, original ONNX flow trajectories for QNN PTQ."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import onnxruntime as ort

from qnn_runtime import ROOT

sys.path.insert(0, str(ROOT / "external/supertonic/py"))
from helper import load_text_processor, load_voice_style


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets", type=Path, default=ROOT / "assets")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/optimization/calibration")
    parser.add_argument("--cases", type=Path, default=ROOT / "scripts/optimization_cases.json")
    parser.add_argument("--latent-length", type=int, default=96)
    parser.add_argument("--steps", type=int, nargs="+", default=[5, 8])
    args = parser.parse_args()
    if not 1 <= args.latent_length <= 128 or any(not 1 <= step <= 100 for step in args.steps) or len(set(args.steps)) != len(args.steps):
        parser.error("Use latent length 1..128 and distinct flow-step counts in 1..100")
    args.output.mkdir(parents=True, exist_ok=True)
    processor = load_text_processor(str(args.assets / "onnx"))
    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    sessions = {name: ort.InferenceSession(str(args.assets / f"onnx/{name}.onnx"), options, providers=["CPUExecutionProvider"])
                for name in ("duration_predictor", "text_encoder", "vector_estimator")}
    manifest = {"source": "original unpadded FP32 ONNX", "cases": [], "text_length": 128, "latent_length": args.latent_length,
                "steps": args.steps, "speed": 1.05, "seed": 193, "complete": False,
                "samples": {"vector_estimator": [], "vocoder": []},
                "asset_manifest": json.loads((args.assets / "manifest.json").read_text())}
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    for index, case in enumerate(json.loads(args.cases.read_text(encoding="utf-8"))["calibration"]):
        ids, text_mask = processor([case["text"]], [case["lang"]])
        style = load_voice_style([str(args.assets / f"voice_styles/{case['voice']}.json")])
        duration = float(sessions["duration_predictor"].run(None, {"text_ids": ids, "style_dp": style.dp, "text_mask": text_mask})[0][0]) / 1.05
        length = int(np.ceil(int(duration * 44100) / 3072))
        if length > args.latent_length or ids.shape[-1] > 128:
            raise ValueError(f"Calibration case exceeds capacity: {case['id']}, {length}")
        embedding = sessions["text_encoder"].run(None, {"text_ids": ids, "style_ttl": style.ttl, "text_mask": text_mask})[0]
        noise = np.random.default_rng(193 + index).standard_normal((1, 144, 128), dtype=np.float32)[..., :length].copy()
        for steps in args.steps:
            latent = noise.copy()
            for step in range(steps):
                feed = {"noisy_latent": latent, "text_emb": embedding, "style_ttl": style.ttl,
                        "latent_mask": np.ones((1, 1, length), np.float32), "text_mask": text_mask,
                        "current_step": np.array([step], np.float32), "total_step": np.array([steps], np.float32)}
                fixed = {}
                for name, value in feed.items():
                    target = args.latent_length if name in ("noisy_latent", "latent_mask") else 128 if name in ("text_emb", "text_mask") else None
                    fixed[name] = np.pad(value, ((0, 0), (0, 0), (0, target - value.shape[-1]))) if target else value
                filename = f"vector_{index:02d}_{steps}_{step:02d}.npz"
                np.savez(args.output / filename, **fixed)
                manifest["samples"]["vector_estimator"].append(filename)
                latent = sessions["vector_estimator"].run(None, feed)[0]
            filename = f"vocoder_{index:02d}_{steps}.npz"
            np.savez(args.output / filename, latent=np.pad(latent, ((0, 0), (0, 0), (0, args.latent_length - length))))
            manifest["samples"]["vocoder"].append(filename)
        manifest["cases"].append({**case, "duration_seconds": duration, "latent_length": length})
        print(case["id"], length, flush=True)
        (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    manifest["complete"] = True
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
