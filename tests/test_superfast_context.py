import json

import pytest

from scripts.build_superfast_context import ROOT, build, sha, validate


@pytest.fixture
def frontend(tmp_path):
    assets = tmp_path / "assets"
    assets.mkdir()
    (assets / "manifest.json").write_text(json.dumps({"files": [
        {"path": "onnx/duration_predictor.onnx", "sha256": "original"}]}))
    program = tmp_path / "duration_predictor.pte"
    program.write_bytes(b"test-program")
    sources = {str(p.relative_to(ROOT)): sha(p) for p in sorted((ROOT / "scripts/superfast").glob("*.py"))}
    sources["scripts/export_superfast_context.py"] = sha(ROOT / "scripts/export_superfast_context.py")
    record = dict(component="duration_predictor", precision="fp16", gelu="exact", profile_level="off",
                  text_length=384, latent_length=128, qnn_sdk="2.50.40.260831", target_soc=88, htp_arch=81,
                  delegates=1, cpu_compute=[], model_sha256="original", pte_sha256=sha(program),
                  pte_bytes=program.stat().st_size, export_sources_sha256=sources,
                  source_transformations_sha256=sha(ROOT / "scripts/superfast/sources.json"),
                  frontend_mode="pack8-norm-tail",
                  inputs=[dict(name=n, shape=s, dtype=d) for n, s, d in [
                      ("text_ids", [1, 384], "int64"), ("style_dp", [1, 8, 16], "float32"),
                      ("text_mask", [1, 1, 384], "float32")]],
                  output=dict(shape=[1], dtype="float32"))
    (tmp_path / "duration_predictor.json").write_text(json.dumps(record))
    return tmp_path, assets, record


def test_resumption_verifies_program_and_compiler_source_before_reuse(frontend):
    directory, assets, record = frontend
    assert validate(directory, "duration_predictor", 128, assets) == record
    (directory / "duration_predictor.pte").write_bytes(b"changed-file")
    with pytest.raises(ValueError, match="Program bytes changed"):
        validate(directory, "duration_predictor", 128, assets)


@pytest.mark.parametrize("key,value", [
    ("text_length", 128), ("cpu_compute", ["aten.add"]), ("qnn_sdk", "another-sdk"),
    ("export_sources_sha256", {}), ("model_sha256", "changed"),
    ("frontend_mode", "different"), ("inputs", []),
])
def test_resumption_rejects_mixed_shapes_sources_or_transformations(frontend, key, value):
    directory, assets, record = frontend
    record[key] = value
    (directory / "duration_predictor.json").write_text(json.dumps(record))
    with pytest.raises(ValueError):
        validate(directory, "duration_predictor", 128, assets)


def test_published_or_incomplete_outputs_are_not_overwritten(tmp_path):
    for name in ("bundle.json", "provenance.json"):
        directory = tmp_path / name
        directory.mkdir()
        (directory / name).write_text("existing")
        with pytest.raises(FileExistsError):
            build(directory, tmp_path, tmp_path)
        assert (directory / name).read_text() == "existing"
    incomplete = tmp_path / "incomplete"
    (incomplete / "frontend").mkdir(parents=True)
    metadata = incomplete / "frontend/duration_predictor.json"
    metadata.write_text("{}")
    with pytest.raises(ValueError):
        build(incomplete, tmp_path, tmp_path)
    assert metadata.read_text() == "{}"
