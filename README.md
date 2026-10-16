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

## Optimization prototypes

The optimization path uses 64-, 96- and 128-frame audio buckets, selected by the native duration predictor. Text models and model files are shared by reference; creating a preset does not duplicate them. Noise is generated once at the base capacity and cropped per channel, preserving the same valid noise across buckets.

```powershell
# Builds both presets using global Python 3.14. --reuse skips matching existing programs.
python scripts/build_optimized.py --reuse

# Conservative: original flow model, reformulated vocoder, automatic buckets.
python scripts/synthesize.py --models artifacts/optimized --steps 8 --compare

# Experimental: faster flow model, but several numerical checks failed.
python scripts/synthesize.py --models artifacts/optimized-fast --steps 8 --compare

# Original 128-frame baseline remains available explicitly.
python scripts/synthesize.py --models artifacts/qnn --steps 8 --compare
```

Both presets use **FP16**, with FP32 public tensors. `ChannelLastConvNeXt` keeps LayerNorm and the pointwise MLP in batch/length/channel order and expresses 1×1 convolutions as linear projections. This preserves the FP32 calculation and the masked/causal boundary rules, while allowing a substantially faster QNN graph. FP16 rounding changes, so this rewrite needs numerical and perceptual evaluation. The conservative preset limits that change to the vocoder.

Bucket selection alone was bit-exact with the baseline audio on all six initial test utterances. The full layout prototype passed the main-sentence screen but later failed short English and Portuguese final-latent limits. It remains experimental. Paired audio is saved alongside each report; these numerical checks do not establish listening quality or word error rate.

Other experiments are available explicitly through `export_qnn.py`, and are not used by either preset:

- `--gelu tanh` expands the standard tanh approximation. It passed numerical checks but did not improve full-pipeline latency in the 96-frame test.
- `--precision a16w8 --calibration ...` applies PTQ with per-channel weights and a min/max activation observer. The tested full-model recipe failed badly on the device, including a calibration utterance. It is unsuitable for use as-is.
- `--quant-scope affine --conv-layout channel-last` restricts PTQ to convolutions and linear projections. This prototype produced 37 delegate partitions with CPU operations remaining, and its eager calibration-sample error was already 10.9%. The exporter rejects remaining CPU compute. The full PTQ vocoder's eager error was 12.4%, before the much larger device error; neither recipe is qualified.

Calibration uses eight separate utterances in five languages, with both five- and eight-step original ONNX trajectories: 104 vector-estimator inputs and 16 vocoder inputs. Generate it with `python scripts/calibrate_qnn.py`. The calibration manifest records the texts, voices, shapes, seed, steps and source assets. `--ptq-only` checks eager PTQ without compiling a program.

Collect detailed operator timing using the already-installed native QNN tools:

```powershell
python scripts/profile_qnn.py --component vector_estimator
python scripts/profile_qnn.py --component vocoder
```

The tool extracts the QNN context, runs `qnn-net-run`, and saves CSV, JSON summaries and output comparisons. It removes the redundant context copy after success unless `--keep-context` is set. Detailed profiling perturbs latency; its node-cycle percentages identify expensive operations and must not be used as synthesis RTF. In the original graphs, GELU accounted for approximately 44% of vector-estimator node cycles and 73% of vocoder node cycles. In the 96-frame layout graphs those shares fell to about 4% and 9%; layout changes were more effective than approximating the activation.

Reproduce the held-out comparison, with no simultaneous compiler or profiler workload:

```powershell
python scripts/qualify_optimizations.py --variant baseline=artifacts/qnn --variant conservative=artifacts/optimized --variant fast=artifacts/optimized-fast --repeat 16 --steps 8 --output artifacts/optimization/final-8steps
```

The numerical screen requires each checked stage's relative L2 error ≤5%, final latent error ≤5%, and spectral convergence ≤10% versus original unpadded FP32 ONNX. These are engineering screening limits, not perceptual acceptance criteria. Reports also include exact baseline comparisons, warm medians/p90, selected buckets, first-request load costs, and program SHA256 hashes. A second seed and five-step runs can be selected with `--seed` and `--steps`. The command collects all cases and exits nonzero when any candidate fails a gate; this is expected for the current experimental fast preset and the long baseline case.

### Optimization results on this device, 2026-10-15

Seven held-out utterances, five languages, seed 42, eight flow steps, speed 1.05, **15 warm requests per case**, Windows Balanced power plan. Models remain loaded within each case; no compiler or detailed profiler runs during these measurements. The unchanged baseline is measured again in the same session, with candidate order reversed on alternating cases.

| Case | Audio | Bucket | Original RTF | Conservative RTF | Experimental fast RTF |
| --- | ---: | ---: | ---: | ---: | ---: |
| English M1 | 5.77 s | 96 | 0.0623 | **0.0414** | 0.0251 |
| Short English F1 | 3.10 s | 64 | 0.1132 | **0.0475** | 0.0344 |
| Korean F2 | 5.41 s | 96 | 0.0602 | **0.0449** | 0.0261 |
| Spanish M3 | 3.51 s | 64 | 0.1038 | **0.0418** | 0.0314 |
| French F4 | 3.32 s | 64 | 0.1050 | **0.0452** | 0.0333 |
| Portuguese M5 | 4.33 s | 64 | 0.0827 | **0.0348** | 0.0227 |
| Longer English M1 | 7.61 s | 128 | 0.0457 | **0.0348** | 0.0214 |

The **conservative preset is recommended for further audio evaluation**: 24–60% lower median latency, all seven median RTFs below 0.05, and bit-identical final flow latents versus the baseline in every case. Its vocoder waveform differs from the baseline by 1.8–2.8% relative L2. Timing is the native synthesis interval, excluding loading, Python preprocessing and file I/O; the highest case p90 is 0.0533 RTF.

Absolute ONNX qualification is **partial**. Six of seven conservative cases pass all screening limits. The longer utterance has 6.03% final-latent error in both the original and conservative paths; this error is inherited from the baseline. The fast preset adds failures on short English (5.50%) and Portuguese (8.17%) at seed 42, and is not recommended as the default. No listening study or ASR/WER evaluation has been performed.

Seed 43 repeats all seven utterances with both presets. Six of seven conservative cases pass; Spanish fails with 7.92% latent error and 14.40% spectral convergence. A separate baseline rerun confirms identical latents and the same inherited failure (14.56% spectral convergence). The fast preset also misses the spectral threshold on Korean at this seed. These seed-sensitive FP16 differences require further accuracy work independently of the speed optimization.

At **five steps**, both original English cases pass the numerical screen with the conservative preset: main sentence **123 ms / 0.0213 RTF**, short sentence **76 ms / 0.0246 RTF**, each with nine warm requests. Matched baseline RTFs are 0.0418 and 0.0877. These results are in `artifacts/optimization/final-5steps/`.

Full measurements and paired WAVs are in `artifacts/optimization/final-8steps/`. The conservative main-sentence audio is `en-main/conservative/supertonic_htp.wav`, beside `onnx_reference.wav`. A [repository evidence snapshot](benchmarks/optimization-2026-10-15.json) retains measurements, failures, profiles, and SHA256 hashes for the tested programs. `scripts/summarize_optimizations.py` regenerates it from the local reports. Keep the explicit original-model command above when comparing future changes.

## Grouped flow experiments

`scripts/export_flow.py` exports 1, 2, 4 or 8 consecutive flow steps as one QNN graph. The initial measurements below use the original convolution layout, exact GELU and FP16; they target a fixed eight-step synthesis schedule. `plainN` only groups calls. `reuseN` also computes text/style attention keys, values and mask-dependent rotary terms once per group. `specialN` additionally precomputes the time encoder and each block's time projection in FP32. The latter can change device rounding and needs separate numerical qualification.

```powershell
# Build one 96-frame bucket; share the existing text models and layout vocoder.
python scripts/build_flow.py --variant reuse8 --bucket 96 --reuse

# Explicit eight-step schedule; the runtime rejects incompatible step counts.
python scripts/synthesize.py --models artifacts/flow/reuse8-buckets96 --steps 8 --compare

# Same bucket and vocoder on both sides; do not compile while measuring.
python scripts/make_bundle.py --bucket artifacts/optimization/fp16-l96 --vocoder artifacts/optimization/layout-l96 --output artifacts/flow/baseline96
python scripts/qualify_optimizations.py --variant baseline=artifacts/flow/baseline96 --variant reuse8=artifacts/flow/reuse8-buckets96 --case en-main --case ko --steps 8 --repeat 16 --output artifacts/flow/screen-reuse8

# Optional timeline of native host calls, copies and cold loads, viewable in Perfetto.
python scripts/synthesize.py --models artifacts/flow/reuse8-buckets96 --steps 8 --repeat 3 --trace --output artifacts/flow/trace-reuse8
```

The exporter checks repeated original unpadded ONNX outputs before conversion and requires exactly one QNN delegate with no CPU compute fallback. The same parameter objects are reused across steps. Metadata records the fixed schedule, reuse options, reference errors, operation counts, compile time and program size. `build_flow.py` accepts `plainN`, `reuseN` and `specialN` for N = 2, 4 or 8, plus `plain1` for layout comparisons; use `--bucket 64 --bucket 96 --bucket 128` only after screening one shape. Each compiled group receives the starting step; the native runner advances by the group size, validates the schedule, and records the actual invocation count.

The compiled 96-frame variants have the following costs. MB is decimal; compilation time excludes model loading and reference checks. Every row compiled into one QNN delegate with no CPU compute fallback.

| Variant | Flow calls per synthesis | Program MB | Compilation seconds |
| --- | ---: | ---: | ---: |
| Original | 8 | 141.30 | — |
| `plain2` | 4 | 143.97 | 169 |
| `reuse2` | 4 | 143.96 | 137 |
| `reuse4` | 2 | 149.28 | 263 |
| `reuse8` | 1 | 160.14 | 653 |
| `special2` | 4 | 143.63 | 128 |
| `special8` | 1 | 160.38 | 589 |

The six-case, seed-42 screen used the same 96-frame bucket and layout vocoder for every variant, with 15 warm requests per case. Median native synthesis times were:

| Case | Original ms | `reuse4` ms | `reuse8` ms | `special8` ms |
| --- | ---: | ---: | ---: | ---: |
| English main | 226.48 | 222.42 | 220.24 | 216.32 |
| English short | 227.36 | 227.25 | 208.27 | 229.90 |
| Korean | 225.95 | 216.63 | 222.56 | 217.83 |
| Spanish | 228.46 | 215.95 | 224.91 | 229.17 |
| French | 221.07 | 228.06 | 231.91 | 225.84 |
| Portuguese | 228.97 | 235.04 | 226.88 | 226.09 |

These results do **not establish a reliable overall speedup**. Repeated screens, including reversed variant order and 31 warm requests, changed the ranking. In the final mixed-bucket comparison, `reuse8` was 17.7% slower on the main English case and 6.7% faster on Korean. An unchanged 64-frame model also varied by 22.9% between the two bundles in that session. The observed variation is large relative to the expected gain. Unrolling removes host crossings and invariant work, but still evaluates all eight flow steps; the existing conservative preset remains recommended. Short-input timings in the table deliberately use length 96 and should not be compared with the faster automatic 64-frame bucket.

`reuse8` produced **bit-identical final latents and audio in all 12 paired cases** across seeds 42 and 43. `reuse4` was also bit-identical in all six seed-42 cases. All six seed-42 cases passed the numerical gates, including `special8`. Seed-43 Spanish failed the existing baseline gates (7.92% final-latent relative L2 and 14.40% spectral convergence); `reuse8` reproduced that failure exactly. Time specialization changed rounding and made this case worse (9.12% and 18.62%). It is retained as an experiment, without a demonstrated consistent latency benefit. These numerical checks are not a listening or intelligibility evaluation; seed-43 runs were accuracy checks with only two warm requests.

An opt-in bundle uses `reuse8` only at length 96 and shares the existing original flow models at 64 and 128. It preserves automatic bucket selection without duplicating binaries:

```powershell
python scripts/make_bundle.py --bucket artifacts/optimization/fp16-l64 --bucket artifacts/flow/reuse8-l96 --bucket artifacts/qnn --vocoder artifacts/optimization/layout-l64 --vocoder artifacts/optimization/layout-l96 --vocoder artifacts/optimization/layout-l128 --output artifacts/flow/reuse8-mixed
python scripts/synthesize.py --models artifacts/flow/reuse8-mixed --steps 8 --compare
```

That mixed bundle produced bit-identical audio to `artifacts/optimized` in all seven held-out cases. Its median RTFs were **0.0328–0.0453**, with 15 warm requests per case. Both bundles retained the known long-English final-latent failure (6.03% relative L2). Only the 96-frame cases use one flow call; 64 and 128 retain eight calls. This bundle is opt-in, not a replacement for the conservative preset.

`host_trace.json` measures **host wall time**, including synchronous backend execution. It is not an HTP hardware-activity trace. The separate HTP Optrace workflow below captures device activity with a matching compiler schematic. Profiling remains disabled for ordinary latency measurements.

The saved `artifacts/flow/trace-baseline96/host_trace.json` and `artifacts/flow/trace-reuse8/host_trace.json` contain 24 and 3 flow-forward spans over three requests, respectively. Open either in [Perfetto](https://ui.perfetto.dev/). Separate detailed QNN profiles in `artifacts/flow/profiles/` still attribute 42.6% of summed node cycles to GELU and 26.4% to permutations in `reuse8`, versus 42.6% and 25.3% in the original 96-frame graph. These instrumented percentages are not production latency; they indicate that the dominant activation/layout work remains after fusion.

The [grouped-flow evidence snapshot](benchmarks/flow-unroll-2026-10-15.json) retains compilation metadata, program hashes, individual warm timings, numerical failures and profile summaries. `python scripts/summarize_flow.py` regenerates it from the local reports without storing model binaries in Git. Adapted attention code retains its upstream [ExecuTorch license](licenses/executorch.txt).

## Convolution-preserving layout experiments

ExecuTorch exports exact GELU as one QNN `Gelu` operator. Its Conv1d lowering introduces `unsqueeze → Conv2d → squeeze`; the rank changes interrupt channels-last propagation. Grouping flow steps does not remove these boundaries inside each ConvNeXt block. Layer normalization and attention also have layout changes, so eliminating every permutation would not be a valid rewrite.

Two narrower alternatives preserve the pointwise convolutions and weights:

- `--conv-layout pointwise-2d` keeps the pair in rank four through GELU, squeezing only after the second convolution. This removes 56 logical transpose nodes per flow step at length 96 (308 → 252), but measured GELU node cycles doubled. Both screened utterances retained bit-identical latents and audio; synthesis was 8–11% slower. This candidate is not recommended.
- `--conv-layout gelu-last` transposes only the activation input from BCL to BLC and transposes its output back. Both pointwise operators remain Conv1d; there is no Conv-to-Linear replacement, GELU approximation, or quantization. The original normalization, masks, padding, residuals and time calculation remain intact.

`plain1` isolates layout from unrolling and still makes eight flow calls. The result is shape-dependent: applying `gelu-last` at length 64 regressed short-English synthesis from 114 ms to 318 ms, despite retaining bit-identical outputs. Use a mixed bundle to select the implementation per bucket. It references existing text models and layout vocoders without copying them:

```powershell
python scripts/build_flow.py --variant plain1 --conv-layout gelu-last --bucket 96 --bucket 128 --work artifacts/pointwise --reuse
python scripts/make_bundle.py --bucket artifacts/optimization/fp16-l64 --bucket artifacts/pointwise/plain1-gelu-last-l96 --bucket artifacts/pointwise/plain1-gelu-last-l128 --vocoder artifacts/optimization/layout-l64 --vocoder artifacts/optimization/layout-l96 --vocoder artifacts/optimization/layout-l128 --output artifacts/pointwise/gelu-last-mixed
python scripts/synthesize.py --models artifacts/pointwise/gelu-last-mixed --steps 8 --compare
```

The mixed bundle uses original flow at 64 frames and `gelu-last` at 96/128. It requires **eight steps** and remains separate from `artifacts/optimized`. The final matched run used seed 42, 15 warm requests per utterance, alternating variant order, and no compiler or profiler workload:

| Case | Bucket | Baseline ms | Mixed ms | Mixed RTF |
| --- | ---: | ---: | ---: | ---: |
| English main | 96 | 187.68 | 153.88 | 0.02667 |
| English short | 64 | 115.23 | 134.18 | 0.04327 |
| Korean | 96 | 192.54 | 172.54 | 0.03190 |
| Spanish | 64 | 120.71 | 113.59 | 0.03238 |
| French | 64 | 113.02 | 113.35 | 0.03416 |
| Portuguese | 64 | 113.36 | 113.14 | 0.02613 |
| English long | 128 | 211.45 | 184.49 | 0.02425 |

The changed 96/128-frame cases improved by **10–18%** in this run. The 64-frame rows use identical binaries; their variation is measurement noise, not an optimization effect. An earlier six-case screen at fixed length 96 found 12–21% reductions. At 128 frames, the initial improvement was only 1.6%, but two repeats with 31 warm requests and opposite variant orders both measured about 183 ms versus 206–208 ms (11–12% faster). Timing is warm native synthesis, excluding model loading, Python preprocessing and file I/O.

All 14 full-bucket comparisons across seeds 42 and 43 produced bit-identical latents and audio to the existing FP16 baseline. The selected mixed bundle was also bit-identical in all seven paired cases. Both paths retain the known long-English seed-42 final-latent error (6.03% relative L2) and Spanish seed-43 errors (7.92% latent relative L2, 14.40% spectral convergence). These exceed the existing absolute ONNX gates; the layout change introduces no additional numerical error in these tests. This is numerical screening, not a listening or intelligibility study.

Combining `gelu-last` with `reuse2` also preserved bit-identical output in the two-case screen. With 31 warm requests, it reduced 157.80 → 154.62 ms on English and 157.51 → 156.26 ms on Korean: only 0.8–2.0%, which does not establish a robust additional gain against the observed variation. It makes four flow calls instead of eight, takes 261 seconds to compile, and produces a 142.94 MB program, versus 108 seconds and 140.80 MB for single-step `gelu-last` at length 96. It remains an optional experiment.

The detailed 96-frame profile retained all 28 GELUs and 86 convolutions. `gelu-last` increased logical transpose nodes from 308 to 364, while reducing their summed measured cycles from 55.23 million to 36.84 million. GELU cycles remained roughly unchanged (92.75 million versus 96.55 million). Operator count alone is therefore a poor cost estimate; tensor layout, tiling and the selected implementation matter. These instrumented cycle totals are not production latency.

The [layout evidence snapshot](benchmarks/pointwise-layout-2026-10-16.json) retains successful and rejected candidates, all timing samples, numerical failures, profiles and program hashes, including the shared models. Regenerate it with `python scripts/summarize_flow.py --work artifacts/pointwise --output benchmarks/pointwise-layout-2026-10-16.json`.

### HTP hardware timeline

Compile a separate profiling program with `--optrace`, then collect it using `qnn-net-run` and the SDK's HTP Optrace reader:

```powershell
python scripts/export_flow.py --unroll 1 --conv-layout gelu-last --latent-length 96 --optrace --output artifacts/pointwise/optrace-gelu-last-l96
python scripts/profile_qnn.py --models artifacts/pointwise/optrace-gelu-last-l96 --component vector_estimator --repeat 3 --optrace --output artifacts/pointwise/profiles/optrace-gelu-last
```

The exporter saves `forward_schematic.bin`. The profiler writes `chrometrace.json`, `chrometrace_runtrace.json`, an HTP graph JSON and an HTML analysis report under `vector_estimator/`. Open the trace in [Perfetto](https://ui.perfetto.dev/) to inspect HVX/HMX/DMA activity and kernel dependencies. The SDK trace uses backend cycles/ticks as coordinates; consult the `Duration (cycles)` and `summary_time_us` attributes for units. These captures use synthetic component inputs and instrumentation changes execution time; ordinary synthesis measurements use separately compiled programs with profiling disabled. Bundle creation rejects profiling programs.

The saved `artifacts/pointwise/profiles/optrace-gelu-last/vector_estimator/chrometrace.json` confirms **separate GELU execution on HVX** in the faster 96-frame graph. Its 28 logical GELUs become 1,792 `q::Gelu.fp16.tcm` kernel tiles, each with shape `[1, 2, 96, 32]` and the `uses_hvx` flag. They are not absorbed into the convolutions. The rank-four `pointwise-2d` capture uses 896 GELU tiles shaped `[1, 1, 96, 128]`; fewer tiles did not make that candidate faster. The capture demonstrates backend tiling and execution, without exposing the kernel's internal arithmetic. Additional detailed profiles at lengths 64 and 128 are archived; per-node cycle totals alone do not explain the 64-frame synthesis regression.

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

All four corrected programs contain **one QNN delegate each and zero CPU compute nodes**. Physical-device checks passed for every component. The current suite has 82 regression tests covering conversion rewrites, masked and causal convolution boundaries, layout equivalence, approximation bounds, grouped-flow equivalence, fixed-schedule validation, and bundle compatibility. Python package dependency checks pass.

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

## Autonomous inference experiments

[research/program.md](research/program.md) adapts Karpathy's autoresearch loop
for controlled TTS inference experiments on O16W. The upstream project remains
unchanged in `external/autoresearch`. Our initial track preserves baseline audio
and final latents exactly, freezes the evaluation inputs, and separates ONNX
checks from repeated native timing rounds. Noisy results are inconclusive.

```powershell
python research/run.py init --campaign artifacts/research/oct16 --baseline artifacts/optimized
python research/run.py run --campaign artifacts/research/oct16 --candidate artifacts/optimized --tag control --description 'Baseline self-comparison'
```

Read the guide before starting a campaign. The runner preserves a baseline
executable, checks model and evaluator hashes, and records decisions in the
campaign's `results.tsv` and per-trial `result.json`. It does not change Git state
or launch an open-ended optimization loop. Existing absolute accuracy failures
remain visible; passing this screen does not establish perceptual speech quality.

## Upstream sources

- [ExecuTorch Supertonic implementation](https://github.com/pytorch/executorch/tree/v1.5.0/examples/models/supertonic), initially contributed by [Young Han (@seyeong-han)](https://github.com/pytorch/executorch/pull/22063).
- [Qualcomm backend documentation](https://docs.pytorch.org/executorch/1.5/backends-qualcomm.html).
- [FluidAudio Supertonic pipeline](https://github.com/FluidInference/FluidAudio/blob/main/Sources/FluidAudio/TTS/Supertonic3/Pipeline/Synthesize/Supertonic3Synthesizer.swift) and [Mobius optimization experiments](https://github.com/FluidInference/mobius/blob/main/models/tts/supertonic-3/OPTIMIZATION.md), reviewed when selecting bucket and graph experiments. Apple measurements are not used as Qualcomm benchmarks.
- Original inference code is the initialized `external/supertonic` submodule.
- [Supertonic 3 assets](https://huggingface.co/supertone-oss-archive/supertonic-3/tree/aafc6e32416a594460b32413efc49d7fe4ce6d46), pinned and recorded in `assets/manifest.json`. See the downloaded `assets/LICENSE` for model terms.
