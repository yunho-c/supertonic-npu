import hashlib
import json
from pathlib import Path

import pytest

from scripts.build_superfast import build


@pytest.fixture
def sources(tmp_path):
    base, flow, vocoder, reference = [tmp_path / n for n in ("frontend", "flow", "vocoder", "reference")]
    for directory in (base, flow, vocoder, reference):
        directory.mkdir()
    specs = {
        "duration_predictor": (base, [["text_ids", [1, 128], "int64"], ["text_mask", [1, 1, 128], "float32"]], [1]),
        "text_encoder": (base, [["text_ids", [1, 128], "int64"], ["text_mask", [1, 1, 128], "float32"]], [1, 256, 128]),
        "vector_estimator": (flow, [["text_emb", [1, 256, 128], "float32"]], [1, 144, 96]),
        "vocoder": (vocoder, [["latent", [1, 144, 96], "float32"]], [1, 1, 294912]),
    }
    for name, (directory, inputs, output) in specs.items():
        data = name.encode()
        (directory / f"{name}.pte").write_bytes(data)
        record = dict(component=name, precision="fp16", gelu="exact", text_length=128,
                      latent_length=96, delegates=1, cpu_compute=[], model_sha256=name,
                      qnn_sdk="sdk", target_soc=88, htp_arch=81,
                      pte_sha256=hashlib.sha256(data).hexdigest(),
                      inputs=[dict(name=n, shape=s, dtype=d) for n, s, d in inputs],
                      output=dict(shape=output, dtype="float32"))
        if name in ("duration_predictor", "text_encoder"):
            record["frontend_mode"] = "pack8-norm-tail"
        elif name == "vector_estimator":
            record.update(flow_total_steps=8, flow_unroll=1, mlp_blocks=list(range(28)),
                          fusion_blocks=list(range(24)), share_guidance_prefix=True, gather_boundary=True)
        else:
            record.update(conv_layout="channel-last", selection=["direct-prelu", "fold-input-affine", "fold-gamma"])
        for target in (directory, reference):
            (target / f"{name}.json").write_text(json.dumps(record))
    (base / "vector_estimator.json").write_text(json.dumps(dict(text_length=128, latent_length=128)))
    for directory in (base, flow, vocoder):
        (directory / "bundle.json").write_text(json.dumps(dict(base=str(base), noise_length=128, flow_total_steps=8,
            buckets=[dict(latent_length=96, models=str(flow), vocoder=str(vocoder), flow_unroll=1, flow_total_steps=8)])))
    return dict(frontend=base, flow=flow, vocoder=vocoder, reference=reference, output=tmp_path / "out")


def test_composed_preset_references_all_four_winners_and_preserves_noise(sources):
    result = build(**sources)
    output = sources["output"]
    assert (output / result["base"]).resolve() == sources["frontend"]
    assert (output / result["buckets"][0]["models"]).resolve() == sources["flow"]
    assert (output / result["buckets"][0]["vocoder"]).resolve() == sources["vocoder"]
    assert result["noise_length"] == 128 and result["flow_total_steps"] == 8
    assert not list(output.glob("*.pte"))
    assert len(json.loads((output / "provenance.json").read_text())["components"]) == 4
    with pytest.raises(FileExistsError):
        build(**sources)


@pytest.mark.parametrize("problem", ["weights", "sdk", "profile", "fallback", "hash", "text_shape", "latent_shape", "partial_pack", "cfg", "steps", "frontend", "vocoder"])
def test_rejects_wrong_or_incompatible_component_before_writing(sources, problem):
    name, directory = "vector_estimator", sources["flow"]
    if problem in ("text_shape", "frontend"):
        name, directory = "text_encoder", sources["frontend"]
    elif problem == "vocoder":
        name, directory = "vocoder", sources["vocoder"]
    path = directory / f"{name}.json"
    data = json.loads(path.read_text())
    if problem == "weights": data["model_sha256"] = "other"
    elif problem == "sdk": data["qnn_sdk"] = "other"
    elif problem == "profile": data["profile_level"] = "optrace"
    elif problem == "fallback": data["cpu_compute"] = ["aten.add"]
    elif problem == "hash": (directory / f"{name}.pte").write_bytes(b"changed")
    elif problem == "text_shape": data["output"]["shape"][-1] = 96
    elif problem == "latent_shape": data["output"]["shape"][-1] = 128
    elif problem == "partial_pack": data["mlp_blocks"] = list(range(24))
    elif problem == "cfg": data["share_guidance_prefix"] = False
    elif problem == "steps": data["flow_total_steps"] = 5
    elif problem == "frontend": data["frontend_mode"] = "packed-compact"
    elif problem == "vocoder": data["selection"].append("fold-head-bn")
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        build(**sources)
    assert not sources["output"].exists()
