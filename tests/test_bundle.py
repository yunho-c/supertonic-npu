import json
import pytest

from scripts.make_bundle import make_bundle


def models(directory, length, text_length=128):
    directory.mkdir()
    for component in ("duration_predictor", "text_encoder", "vector_estimator", "vocoder"):
        (directory / f"{component}.json").write_text(json.dumps({
            "latent_length": length, "text_length": text_length, "cpu_compute": [], "model_sha256": component}))
        (directory / f"{component}.pte").touch()
    return directory


def test_bundle_shares_base_and_applies_vocoder_override_without_copying(tmp_path):
    base = models(tmp_path / "base", 128)
    small = models(tmp_path / "small", 64)
    vocoder = models(tmp_path / "vocoder", 64)
    out = tmp_path / "bundle"
    make_bundle(base, [base, small], out, [vocoder])
    bundle = json.loads((out / "bundle.json").read_text())
    assert [entry["latent_length"] for entry in bundle["buckets"]] == [64, 128]
    assert (out / bundle["base"]).resolve() == base
    assert (out / bundle["buckets"][0]["vocoder"]).resolve() == vocoder
    assert list(out.glob("*.pte")) == []


@pytest.mark.parametrize("problem", ["text", "vocoder", "missing", "fallback", "hash"])
def test_bundle_rejects_incompatible_or_incomplete_programs(tmp_path, problem):
    base = models(tmp_path / "base", 128)
    candidate = models(tmp_path / "candidate", 64)
    metadata_path = candidate / "vector_estimator.json"
    metadata = json.loads(metadata_path.read_text())
    if problem == "text":
        metadata["text_length"] = 64
    elif problem == "vocoder":
        metadata["latent_length"] = 32
    elif problem == "missing":
        (candidate / "vector_estimator.pte").unlink()
    elif problem == "fallback":
        metadata["cpu_compute"] = ["aten.add"]
    elif problem == "hash":
        metadata["model_sha256"] = "wrong"
    metadata_path.write_text(json.dumps(metadata))
    with pytest.raises((ValueError, FileNotFoundError)):
        make_bundle(base, [candidate], tmp_path / "bundle")
