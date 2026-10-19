# Experimental T384 Superfast

`scripts/build_superfast_context.py` builds a separate FP16 preset for MLSR's
duration-planning API: 384 normalized text tokens, VE/vocoder pairs at latent
lengths 128, 256 and 512, and a fixed source-noise capacity of 512. The largest
bucket supports 1,572,864 PCM frames (about 35.7 seconds at 44.1 kHz).

```powershell
python scripts/build_superfast_context.py --output artifacts/superfast-t384-<new-run>
python scripts/synthesize.py --models artifacts/superfast-t384-<new-run> --steps 8 --text "A complete sentence to synthesize."
```

Use the repository's pinned ExecuTorch source, original assets and QAIRT
2.50.40.260831 environment. The target remains Windows ARM64 Snapdragon X2 Elite,
SoC 88 / HTP V81. Other targets require separately compiled and qualified models.
Compilation is sequential. Logs and incomplete evidence are retained; completed
components can be resumed only if their bytes, source and tensor contracts match.
A published preset cannot be overwritten. `--assemble-only` checks existing
components without compiling.

The frozen modules in `scripts/superfast` preserve the measured transformations
and their exact source hashes. The frontend uses pack8/norm-tail; all 28 flow
MLPs are packed, the first 24 retain the norm-tail fusion, and CFG shares its
prefix. Exact GELU and eight separate HTP FP16 updates are retained. The vocoder
uses direct PReLU and folded input affine/gamma. Each graph must contain one QNN
delegate and zero CPU compute operations.

Every export checks the FP32 transformed graph against the original ONNX before
compilation. This catches transformation errors; it does not establish device
accuracy or subjective audio quality. MLSR performs device and prepared-duration
qualification separately. Listening remains a user evaluation.

`bundle.json` selects the smallest fitting bucket. Its base is the real L512
pair, with verified copies of the frontend, so existing native tools infer the
correct 512-position source noise. The old `superfast` preset and its T128/L96
artifacts remain independent. Seeded PCM across the two presets is expected to
differ because both the context and the noise layout change.
