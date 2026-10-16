"""Synthesize speech on HTP, with optional stage and end-to-end ONNX comparison."""

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import onnxruntime as ort
import soundfile as sf

from qnn_runtime import ROOT, NativeRunner, metrics, tensor_file

sys.path.insert(0, str(ROOT / "external/supertonic/py"))
from helper import load_text_processor, load_voice_style


def spectrum(audio):
    frames = np.lib.stride_tricks.sliding_window_view(np.pad(audio, (1024, 1024)), 2048)[::512]
    return np.abs(np.fft.rfft(frames * np.hanning(2048)))


def compare_onnx(args, inputs, timing, report, sample_rate, output):
    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    sessions = {name: ort.InferenceSession(str(args.assets / "onnx" / f"{name}.onnx"), sess_options=options, providers=["CPUExecutionProvider"])
                for name in ("duration_predictor", "text_encoder", "vector_estimator", "vocoder")}
    stage_timings = {}

    def run(name, feed):
        start = time.perf_counter()
        result = sessions[name].run(None, feed)[0]
        stage_timings.setdefault(name, []).append((time.perf_counter() - start) * 1000)
        return result

    text_size = int(inputs["text_mask"].sum())
    latent_size = timing["latent_length"]
    reference_text = {"text_ids": inputs["text_ids"][..., :text_size],
                      "text_mask": inputs["text_mask"][..., :text_size]}
    duration = run("duration_predictor", {**reference_text, "style_dp": inputs["style_dp"]})
    embedding = run("text_encoder", {**reference_text, "style_ttl": inputs["style_ttl"]})
    actual_duration = np.fromfile(output / "duration.raw", np.float32).reshape(duration.shape)
    actual_embedding = np.fromfile(output / "text_emb.raw", np.float32).reshape(1, 256, inputs["text_mask"].shape[-1])[..., :text_size]
    report["real_input_stages"] = {"duration_predictor": metrics(actual_duration, duration), "text_encoder": metrics(actual_embedding, embedding)}
    # Match the HTP duration-derived mask to isolate numerical inference error.
    latent = inputs["noise"][..., :latent_size].copy()
    mask = np.ones((1, 1, latent_size), np.float32)
    for step in range(args.steps):
        latent = run("vector_estimator", {"noisy_latent": latent, "text_emb": embedding,
            "style_ttl": inputs["style_ttl"], "latent_mask": mask, "text_mask": reference_text["text_mask"],
            "current_step": np.array([step], np.float32), "total_step": np.array([args.steps], np.float32)})
    baseline = run("vocoder", {"latent": latent}).ravel()[:timing["wav_length"]]
    actual_latent = np.fromfile(output / "latent.raw", np.float32).reshape(inputs["noise"].shape)[..., :latent_size]
    actual_audio = np.fromfile(output / "audio.raw", np.float32).ravel()[:timing["wav_length"]]
    isolated_vocoder = sessions["vocoder"].run(None, {"latent": actual_latent})[0].ravel()[:timing["wav_length"]]
    report["real_input_stages"]["vocoder"] = metrics(actual_audio, isolated_vocoder)
    report["flow_end_to_end"] = metrics(actual_latent, latent)
    report["waveform_end_to_end"] = metrics(actual_audio, baseline)
    expected_spectrum, actual_spectrum = spectrum(baseline), spectrum(actual_audio)
    report["spectral_convergence"] = float(np.linalg.norm(actual_spectrum - expected_spectrum) / max(np.linalg.norm(expected_spectrum), 1e-30))
    report["onnx_cpu_x64_emulated_stage_ms"] = stage_timings
    report["onnx_predicted_seconds"] = float(duration[0] / args.speed)
    report["comparison_mask"] = "Original unpadded ONNX; HTP predicted duration, same noise, voice and flow steps"
    report["padding_handling"] = "HTP ConvNeXt padding replicates the last valid masked position"
    sf.write(output / "onnx_reference.wav", baseline, sample_rate, subtype="PCM_16")
    for name, comparison in report["real_input_stages"].items():
        if comparison["relative_l2"] > 0.05:
            raise AssertionError(f"{name}: real-input relative L2 error exceeds 5%")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--text", default="Hello! This is Supertonic running on the Qualcomm neural processing unit.")
    parser.add_argument("--lang", default="en")
    parser.add_argument("--voice", default="M1")
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--speed", type=float, default=1.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--compare", action="store_true")
    parser.add_argument("--assets", type=Path, default=ROOT / "assets")
    parser.add_argument("--models", type=Path, default=ROOT / "artifacts/qnn")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/speech")
    args = parser.parse_args()
    if args.repeat < 1 or not 1 <= args.steps <= 100 or not np.isfinite(args.speed) or args.speed <= 0:
        parser.error("repeat, steps and speed must be positive (steps <= 100)")
    args.output.mkdir(parents=True, exist_ok=True)
    config = json.loads((args.assets / "onnx/tts.json").read_text())
    metadata = json.loads((args.models / "vector_estimator.json").read_text())
    text_length, latent_length = metadata["text_length"], metadata["latent_length"]
    processor = load_text_processor(str(args.assets / "onnx"))
    text_ids, text_mask = processor([args.text], [args.lang])
    if text_ids.shape[-1] > text_length:
        parser.error(f"Text needs {text_ids.shape[-1]} tokens; compiled bucket supports {text_length}")
    padding = text_length - text_ids.shape[-1]
    style = load_voice_style([str(args.assets / "voice_styles" / f"{args.voice}.json")])
    inputs = {"text_ids": np.pad(text_ids, ((0, 0), (0, padding))),
              "text_mask": np.pad(text_mask, ((0, 0), (0, 0), (0, padding))),
              "style_dp": style.dp, "style_ttl": style.ttl,
              "noise": np.random.default_rng(args.seed).standard_normal((1, config["ttl"]["latent_dim"] * config["ttl"]["chunk_compress_factor"], latent_length), dtype=np.float32)}
    specs = {key: tensor_file(args.output / "inputs", key, value) for key, value in inputs.items()}
    sample_rate = config["ae"]["sample_rate"]
    request = {"command": "synthesize", "inputs": specs, "output_dir": str(args.output.resolve()),
               "speed": args.speed, "steps": args.steps, "sample_rate": sample_rate,
               "samples_per_latent": config["ae"]["base_chunk_size"] * config["ttl"]["chunk_compress_factor"]}
    report = {"text": args.text, "lang": args.lang, "voice": args.voice, "seed": args.seed,
              "steps": args.steps, "speed": args.speed, "sample_rate": sample_rate, "runs": []}
    with NativeRunner(args.models, args.output / "htp.log") as runner:
        for i in range(args.repeat):
            timing = runner.request(request)
            report["runs"].append(timing)
            print(f"Run {i + 1}: {timing['duration_seconds']:.2f}s audio in {timing['inference_ms']:.1f}ms; RTF {timing['rtf']:.4f}", flush=True)
    warm = report["runs"][1:]
    if warm:
        report["warm_summary"] = {
            "count": len(warm),
            "synthesis_ms_median": float(np.median([run["inference_ms"] for run in warm])),
            "synthesis_ms_p90": float(np.percentile([run["inference_ms"] for run in warm], 90)),
            "rtf_median": float(np.median([run["rtf"] for run in warm])),
            "stage_ms_median": {name: float(np.median([sum(run[name + "_ms"]) for run in warm]))
                                for name in ("duration_predictor", "text_encoder", "vector_estimator", "vocoder")},
        }
    audio = np.fromfile(args.output / "audio.raw", np.float32)[:timing["wav_length"]]
    if not np.isfinite(audio).all() or np.max(np.abs(audio)) < 1e-5:
        raise AssertionError("Output audio is non-finite or silent")
    sf.write(args.output / "supertonic_htp.wav", audio, sample_rate, subtype="PCM_16")
    if args.compare:
        try:
            compare_onnx(args, inputs, timing, report, sample_rate, args.output)
        finally:
            (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    else:
        (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
