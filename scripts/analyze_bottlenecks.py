"""Read saved Supertonic measurements and derive an arithmetic/traffic cost model.

This does not compile models, run HTP benchmarks, or change inference programs.
--probe-algebra checks proposed identities in eager FP32 with the published weights.
"""

import argparse
import hashlib
import json
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]


def read(relative):
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


def cost_model(length, text_length=128):
    config = read("assets/onnx/tts.json")
    field = config["ttl"]["vector_field"]
    main = field["main_blocks"]
    blocks = main["n_blocks"]
    count = blocks * sum(main[key]["num_layers"] for key in
                         ("convnext_0", "convnext_1", "convnext_2")) + field["last_convnext"]["num_layers"]
    channels = main["convnext_0"]["idim"]
    expanded = main["convnext_0"]["intermediate_dim"]
    text_channels = main["text_cond_layer"]["text_dim"]
    style_channels = main["style_cond_layer"]["style_dim"]
    style_tokens = config["ttl"]["style_encoder"]["style_token_layer"]["n_style"]
    latent_channels = config["ttl"]["latent_dim"] * config["ttl"]["chunk_compress_factor"]
    batch = 2  # The conditional and unconditional branches of CFG.
    mac = {
        "pointwise_pairs": count * batch * length * 2 * channels * expanded,
        "depthwise": count * batch * length * channels * 5,
        "input_output_projections": 2 * batch * length * latent_channels * channels,
        "text_query_output": blocks * 2 * batch * length * channels * channels,
        "text_key_value": blocks * 2 * batch * text_length * text_channels * channels,
        "text_attention_products": blocks * 2 * batch * length * text_length * channels,
        "style_query_output": blocks * 2 * batch * length * channels * style_channels,
        "style_key_value": blocks * 2 * batch * style_tokens * style_channels * style_channels,
        "style_attention_products": blocks * 2 * batch * length * style_tokens * style_channels,
        "time_encoder": batch * 2 * field["time_encoder"]["time_dim"] * field["time_encoder"]["hdim"],
        "time_projections": blocks * batch * field["time_encoder"]["time_dim"] * channels,
    }
    decoder = config["ae"]["decoder"]
    vocoder_length = length * config["ttl"]["chunk_compress_factor"]
    vocoder_mac = vocoder_length * (
        config["ae"]["ldim"] * decoder["hdim"] * decoder["ksz_init"]
        + decoder["num_layers"] * (2 * decoder["hdim"] * decoder["intermediate_dim"]
                                   + decoder["hdim"] * decoder["ksz"])
        + decoder["head"]["idim"] * decoder["head"]["hdim"] * decoder["head"]["ksz"]
        + decoder["head"]["hdim"] * decoder["head"]["odim"])
    prefix_count = main["convnext_0"]["num_layers"] + main["convnext_1"]["num_layers"]
    prefix_mac_saved = length * (latent_channels * channels
                                + prefix_count * (2 * channels * expanded + 5 * channels))
    return {
        "latent_length": length, "text_length": text_length, "cfg_batch": batch,
        "convnext_blocks": count, "flow_mac_by_category": mac,
        "flow_mac_per_step": sum(mac.values()), "flow_flops_per_step": 2 * sum(mac.values()),
        "pointwise_mac_fraction": mac["pointwise_pairs"] / sum(mac.values()),
        "vocoder_mac": vocoder_mac, "vocoder_flops": 2 * vocoder_mac,
        "gelu_logical_fp16_bytes_one_direction": count * batch * length * expanded * 2,
        "latent_fp32_bytes": latent_channels * length * 4,
        "pointwise_fp16_weight_bytes": count * 2 * channels * expanded * 2,
        "pointwise_weight_only_flops_per_byte": batch * length,
        "shared_prefix_convnext_blocks": prefix_count,
        "shared_prefix_mac_saved": prefix_mac_saved,
        "shared_prefix_mac_fraction_saved": prefix_mac_saved / sum(mac.values()),
        "qualification": "Analytic dense MACs before constant folding; excludes elementwise arithmetic, reductions, padding, transfers, and compiler tiling. One MAC counts as two FLOPs.",
    }


def hardware_trace(relative, logical_gelu_bytes):
    path = ROOT / relative
    data = json.loads(path.read_text())["data"]
    overall = data["htp_overall_summary"]["data"][0]
    provenance = json.loads((path.parent / "provenance.json").read_text())
    gelu = next(row for row in data["qnn_op_types"]["data"] if row["op"] == "Gelu")
    return {
        "source": relative, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "model_pte_sha256": provenance["pte_sha256"],
        "layout": provenance["metadata"]["conv_layout"],
        "overall": overall,
        "qnn_dominant_path": sorted(data["qnn_op_types"]["data"],
                                     key=lambda row: -row["num_dominant_path_cycles_htp_0"]),
        "htp_dominant_path": sorted(data["htp_op_types"]["data"],
                                     key=lambda row: -row["num_dominant_path_cycles_htp_0"]),
        "gelu_read_amplification": gelu["vtcm_read"] / logical_gelu_bytes,
        "gelu_write_amplification": gelu["vtcm_write"] / logical_gelu_bytes,
        "qualification": "SDK-reported traffic and dominant path from an instrumented synthetic-input trace; not production latency, peak-bandwidth utilization, or a hardware byte-counter measurement.",
    }


def measurements():
    output = {}
    summaries = [ROOT / "artifacts/optimization/final-8steps/summary.json"]
    summaries += sorted((ROOT / "artifacts/pointwise").glob("*/summary.json"))
    for path in summaries:
        source = json.loads(path.read_text())
        rows = []
        for case, variants in source.get("cases", {}).items():
            for name, variant in variants.items():
                rows.append({"case": case, "variant": name,
                             "warm": variant.get("warm"),
                             **{k: variant.get(k) for k in ("bucket_length", "bit_exact_with_baseline",
                                                            "numerical_gates_pass", "flow_relative_l2")}})
        output[str(path.relative_to(ROOT))] = {
            "seed": source.get("seed"), "steps": source.get("steps"),
            "warm_requests_per_case": source.get("warm_requests_per_case"), "rows": rows}
    return output


def algebra_probes():
    sys.path[:0] = [str(ROOT / ".deps"), str(ROOT / ".deps/executorch")]
    import torch
    from torch.nn import functional as F
    from examples.models.supertonic.loaders.checkpoint_loader import load_vector_estimator
    from examples.models.supertonic.model.config import TTSConfig
    from examples.models.supertonic.source_transformations.mlx import replace_same_padding
    from qnn_transformations import replace_masked_padding

    torch.set_num_threads(2)
    torch.manual_seed(20261016)
    config = TTSConfig.from_json(ROOT / "assets/onnx/tts.json")
    model = load_vector_estimator(ROOT / "assets/onnx/vector_estimator.onnx", config).eval()
    model = replace_masked_padding(replace_same_padding(model))
    field = model.vector_field
    results = {"qualification": "Eager FP32 algebra checks with published weights. No QNN export, device equivalence, perceptual quality, or latency claim."}

    def error(actual, expected):
        delta = (actual - expected).double()
        return {"max_abs": delta.abs().max().item(),
                "relative_l2": (delta.norm() / expected.double().norm().clamp_min(1e-30)).item(),
                "bit_exact": torch.equal(actual, expected)}

    with torch.inference_mode():
        results["parameters"] = sum(p.numel() for p in model.parameters())
        results["packed_pointwise"] = []
        block = field.main_blocks[0].convnext[0].block
        for length in (64, 96, 128):
            x = torch.randn(2, 512, length)
            expected = block.pwconv2(block.act(block.pwconv1(x)))
            packed = x.transpose(1, 2).reshape(1, 8, 2 * length // 8, 512).permute(0, 3, 1, 2)
            y = F.conv2d(packed, block.pwconv1.weight.unsqueeze(-1), block.pwconv1.bias)
            y = F.conv2d(block.act(y), block.pwconv2.weight.unsqueeze(-1), block.pwconv2.bias)
            y = y.permute(0, 2, 3, 1).reshape(2, length, 512).transpose(1, 2)
            results["packed_pointwise"].append({"length": length, **error(y, expected)})
            torch.testing.assert_close(y, expected, atol=2e-4, rtol=2e-4)

        results["constant_unconditional_text_attention"] = []
        for index in range(4):
            attn = field.main_blocks[index * 6 + 3].attn
            for valid_text in (1, 73, 128):
                x = torch.randn(1, 96, 512)
                qm = torch.ones(1, 1, 96); qm[..., 83:] = 0
                km = torch.ones(1, 1, 128); km[..., valid_text:] = 0
                token = model.uncond_masker.text_special_token.transpose(1, 2)
                context = token.expand(1, 128, 256)
                expected = attn(x, context, qm, km)
                y = attn.out_fc(attn.W_value(token)).expand(1, 96, 512) * qm.transpose(1, 2)
                results["constant_unconditional_text_attention"].append(
                    {"block": index, "valid_text": valid_text, **error(y, expected)})
                torch.testing.assert_close(y, expected, atol=2e-4, rtol=2e-4)

        results["shared_prefix"] = []
        for length, valid_length in ((64, 45), (96, 83), (128, 110)):
            x = torch.randn(1, 144, length)
            mask = torch.ones(1, 1, length); mask[..., valid_length:] = 0
            def prefix(value, lm, time):
                h = field.proj_in(value) * lm
                h = field.main_blocks[0](h, lm)
                h = field.main_blocks[1](h, field.time_encoder(time), lm)
                return field.main_blocks[2](h, lm)
            expected = prefix(x.repeat(2, 1, 1), mask.repeat(2, 1, 1), torch.full((2,), 0.375))
            y = prefix(x, mask, torch.tensor([0.375])).repeat(2, 1, 1)
            results["shared_prefix"].append({"length": length, **error(y, expected)})
            torch.testing.assert_close(y, expected, atol=2e-4, rtol=2e-4)

        x = torch.randn(2, 512, 96)
        mask = torch.ones(2, 1, 96); mask[..., 83:] = 0
        residual = x * mask
        terminal = mask - torch.cat((mask[..., 1:], torch.zeros_like(mask[..., :1])), -1)
        expected = (residual * terminal).sum(-1, keepdim=True)
        results["last_valid_gather"] = error(residual[..., 82:83], expected)
        assert torch.equal(residual[..., 82:83], expected)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "benchmarks/first-principles-2026-10-16.json")
    parser.add_argument("--probe-algebra", action="store_true")
    args = parser.parse_args()
    costs = {str(length): cost_model(length) for length in (64, 96, 128)}
    traces = {}
    for variant in ("optrace-pointwise", "optrace-gelu-last"):
        path = f"artifacts/pointwise/profiles/{variant}/vector_estimator/chrometrace_qnn_htp_analysis_summary.json"
        if (ROOT / path).is_file():
            traces[variant] = hardware_trace(path, costs["96"]["gelu_logical_fp16_bytes_one_direction"])
    report = read("artifacts/pointwise/qualification-seed42/en-main/gelu_last/report.json")
    warm_runs = report["runs"][1:]
    overhead = [run["inference_ms"] - sum(sum(run[f"{component}_ms"]) for component in
                 ("duration_predictor", "text_encoder", "vector_estimator", "vocoder")) for run in warm_runs]
    result = {"device": "Snapdragon X2 Elite X2E84100 / HTP V81", "date": "2026-10-16",
              "costs": costs, "traces": traces, "measurements": measurements(),
              "host_outside_forward_ms_median": statistics.median(overhead),
              "host_overhead_note": "Excludes backend/RPC/buffer overhead inside Module.forward; paired per-request subtraction.",
              "algebra_probes": algebra_probes() if args.probe_algebra else None}
    micro = ROOT / "artifacts/first-principles/pointwise-probe/summary.json"
    if micro.is_file():
        result["pointwise_probe"] = json.loads(micro.read_text())
        result["packed_vs_linear_bit_exact"] = {
            str(length): (micro.parent / f"packed8-l{length}/reverse.raw").read_bytes()
            == (micro.parent / f"linear-l{length}/reverse.raw").read_bytes()
            for length in (64, 96, 128)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(args.output)
    print(json.dumps({"cost96": costs["96"], "host_ms": result["host_outside_forward_ms_median"],
                      "algebra": result["algebra_probes"]}, indent=2))


if __name__ == "__main__":
    main()
