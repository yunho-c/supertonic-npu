# Supertonic 3 on Snapdragon X2 Elite HTP

This project reconstructs Supertonic 3's four PyTorch components from the published ONNX weights, exports them through ExecuTorch's Qualcomm backend, and runs the resulting `.pte` programs in a native Windows ARM64 process. Python handles text preparation, test orchestration and WAV output. The native process keeps the models loaded and executes the flow loop on HTP.

The target is the **Snapdragon X2 Elite X2E84100**, using SoC ID 88 / HTP V81. This target addition is specific to this machine; other Snapdragon generations require their own compiler configuration.

## Environment

- Global Python **3.14.7**, with no virtual environment. The installed interpreter is x64 under Windows ARM emulation; the inference executable is native ARM64.
- ExecuTorch **1.5.0**, source revision `f7140a46ff38e919c557d45b102d3ff26097c8c9`.
- PyTorch **2.14.0**, matching ExecuTorch's `torch_pin.py`.
- QAIRT **2.50.40.260831**, already installed at `C:\Qualcomm\AIStack\QAIRT\2.50.40.260831`.
- Visual Studio 2022 Build Tools with ARM64 C++ tools, CMake, and Git.

## Build and run

From this directory in PowerShell:

```powershell
# Reuses the global Python installation and existing QAIRT SDK.
./scripts/bootstrap.ps1

# FP16 HTP compute, float32 public inputs/outputs, fixed batch size 1.
python scripts/export_qnn.py

# Rejects exports containing CPU compute; compares physical HTP outputs to ONNX.
python scripts/verify_qnn.py

# Runs one cold and two warm requests and writes WAVs and timing/error reports.
python scripts/synthesize.py --compare
```

Example with different text and voice:

```powershell
python scripts/synthesize.py --text 'Hello from the Qualcomm NPU.' --voice F1 --output artifacts/hello
```

Output is `artifacts/speech/supertonic_htp.wav` by default. `--compare` also creates `onnx_reference.wav` and records numerical comparisons. The ONNX timing is from x64 CPU execution under emulation and must not be presented as a native CPU speedup comparison.

Downloads go directly into `assets/`, with pinned revisions and checksum verification. Pip uses `--no-cache-dir`. Only the required ExecuTorch submodules are initialized. Models, build trees and dependencies are ignored by Git.

For individual build steps:

```powershell
python scripts/prepare_executorch.py
./scripts/build-executorch.ps1 -Target Host
./scripts/build-executorch.ps1 -Target ARM64
./scripts/build-runner.ps1
python scripts/export_qnn.py --component vector_estimator
python -m pytest tests -q
```

## Conversion changes

`prepare_executorch.py` adds SC8480XP / V81 to ExecuTorch's Python and FlatBuffer target schemas. It requests an 8 MB VTCM allocation; this is a compiler setting, not a statement about total hardware capacity.

The exporter reuses upstream's padding and relative-attention transformations. Additional changes in `qnn_transformations.py` preserve behavior and complete delegation:

1. Boolean `bitwise_not` becomes `logical_not`, which the QNN backend supports. Integer inversion is explicitly rejected.
2. Softplus uses the stable equivalent `relu(x) + log1p(exp(-abs(x)))`. This avoids a scalar comparison introduced too late for QNN's scalar-lifting pass.
3. Masked ConvNeXt blocks replicate the last **valid** position before convolution. Simply appending zeros changes Supertonic's boundary behavior; this correction makes fixed buckets match the original unpadded model for shorter inputs.

Every transformed model is compared with the original unpadded ONNX model before conversion, including shorter text and latent sequences inside a larger bucket. Export fails if any compute remains outside QNN delegates. Each component's `.json` and `_graph.txt` record the actual exported shapes and delegation.

## Scope and measurements

The default build uses **128 text positions and 128 latent positions**, batch size 1. Text positions include the language tags added by Supertonic's preprocessing. At 44,100 Hz and 3,072 samples per latent position, the speech capacity is about **8.92 seconds**. Overlong text or predicted speech is rejected rather than truncated. Longer content needs chunking or larger compiled buckets; these scripts do not implement automatic chunking.

Use `--text-length`, `--latent-length` and a separate `--output` directory when exporting another shape. All four components in a directory must use compatible shapes. The native runner accepts 1–100 flow steps; the tested default is 5. Voices and language preprocessing come from the original Supertonic submodule.

`verify_qnn.py` compares each component's valid positions with the original unpadded ONNX output. `synthesize.py --compare` additionally checks real text/voice inputs, the final latent, and the resulting waveform. The end-to-end comparison uses the HTP-predicted duration on both paths to isolate inference error. It also records the independent ONNX duration estimate. These checks are numerical validation, not a perceptual speech-quality benchmark.

Timing reports separate first-load cost, each component invocation, the five vector-estimator calls, and the native synthesis interval. The synthesis interval excludes model loading and file writes; `request_ms` includes both. Compare warm runs when assessing steady-state inference.

### Validated on this device, 2026-10-15

All four corrected programs contain **one QNN delegate each and zero CPU compute nodes**. Physical-device checks passed for every component. Nine regression tests cover the conversion rewrites and convolution boundaries, including length-one sequences and dilated convolutions. Python package dependency checks pass.

| Speech case | Audio duration | Warm synthesis median | Real-time factor |
| --- | ---: | ---: | ---: |
| English, M1, default sentence | 5.77 s | 263 ms (9 warm runs) | 0.0456 |
| English, F1, quick brown fox | 3.10 s | 279 ms (2 warm runs) | 0.0898 |
| Korean, F2 | 5.41 s | 293 ms (2 warm runs) | 0.0541 |

The main case's first native request took **718 ms**, including model loading and output writes; process startup is outside that interval. Warm p90 was **288 ms**. This is about **22 times real time** for that sentence, not a claim of a 22× speedup over a CPU implementation.

On the main case, median stage totals were 3.48 ms for duration prediction, 4.17 ms for text encoding, 163 ms for five vector-estimator calls, and 99.5 ms for the vocoder. These independently calculated medians need not add to the median synthesis time. The flow loop and vocoder are the main optimization targets. The sentence needs 83 latent positions while the compiled bucket processes 128; smaller shape buckets are a concrete next optimization. Quantization would require representative calibration and further audio-quality evaluation.

FP16 is not bit-exact with float32 ONNX. Across these three real speech cases, text-embedding relative L2 error was 0.195–0.246%; isolated vocoder error was 0.897–1.24%; final-latent error was 1.43–4.38%. End-to-end waveform cosine similarity was 0.913–0.988, with spectral convergence error 3.25–8.91%. The waveform differences remain a limitation; no perceptual listening study or full language/voice quality benchmark has been performed.

The exact measurements and paired audio are in:

- `artifacts/qnn/verification/report.json`
- `artifacts/speech/report.json`, `supertonic_htp.wav`, and `onnx_reference.wav`
- `artifacts/speech-f1/` and `artifacts/speech-ko/`

Reproduce the main measurement with `python scripts/synthesize.py --compare --repeat 10`.

## Upstream sources

- [ExecuTorch Supertonic implementation](https://github.com/pytorch/executorch/tree/v1.5.0/examples/models/supertonic), initially contributed by [Young Han (@seyeong-han)](https://github.com/pytorch/executorch/pull/22063).
- [Qualcomm backend documentation](https://docs.pytorch.org/executorch/1.5/backends-qualcomm.html).
- Original inference code is the initialized `external/supertonic` submodule.
- [Supertonic 3 assets](https://huggingface.co/supertone-oss-archive/supertonic-3/tree/aafc6e32416a594460b32413efc49d7fe4ce6d46), pinned and recorded in `assets/manifest.json`. See the downloaded `assets/LICENSE` for model terms.
