# Supertonic 3 inference autoresearch

Adapted from the experiment loop in `external/autoresearch/program.md` (Karpathy,
MIT). Keep that submodule unchanged. Run this workflow in the **supertonic-npu
repository**, not inside the upstream training project.

## Objective

Reduce warm native synthesis latency on O16W's Snapdragon X2 Elite HTP V81 while
preserving Supertonic 3's output. Start with the existing conservative bundle,
`artifacts/optimized`. The Python ONNX reference runs under x64 emulation: use it
for accuracy, never as a native CPU speedup claim.

This first protocol is **exact-execution**: weights, FP16 precision, exact GELU,
text preprocessing, voices, flow steps, speech speed and input noise are fixed.
Candidate final latents and waveform samples must equal the baseline exactly on
every tested case and seed. This is deliberately restrictive. Different rounding,
quantization, approximate activations, fewer flow steps, distillation and model
changes need a separately designed quality protocol with listening and
intelligibility evaluation. Do not silently weaken this protocol to admit them.

## Read before experimenting

- `README.md`, especially optimization evidence, grouped flow, and timing noise.
- `research/protocol.json` and `research/run.py`, the fixed evaluation contract.
- `scripts/qualify_optimizations.py`, `scripts/synthesize.py`,
  `scripts/qnn_runtime.py`, and `scripts/optimization_cases.json`.
- Relevant exporters, transformations, native runtime, and their tests.
- `benchmarks/optimization-2026-10-15.json` and any later evidence snapshots.

The optimization cases have already informed this project's development. They
are a regression corpus, not a fresh unseen test set. Reserve additional texts,
voices and seeds for later confirmation before claiming general performance.

## Set up a campaign

Use an isolated branch/worktree with an intentional, reproducible starting
snapshot. Inspect `git status` first. Never reset, stash, commit or delete someone
else's pending work merely to prepare an experiment. Existing uncommitted work
in the main checkout must be preserved; do not assume a worktree at HEAD contains
it. The runner records the commit, dirty patch, and copies of implementation
sources, including untracked files, but does not manage branches or commits.

Provision the isolated checkout using this repository's normal build/asset
instructions. Each checkout needs its own native build and experiment outputs.
Keep baseline artifacts immutable; candidates may reference unchanged baseline
files, but must write modified programs to their own directories.

When asked to start research, use the user's requested experiment/time budget.
If none was specified, run a bounded batch of at most 10 candidates or two hours,
whichever comes first, then report results. Finish recording the current trial
when the budget ends. An instruction to update this workflow is not an instruction
to start an unattended optimization campaign.

From the repository root in PowerShell:

```powershell
python research/run.py init --campaign artifacts/research/oct16 --baseline artifacts/optimized

# First establish a control: compare the baseline to itself.
python research/run.py run --campaign artifacts/research/oct16 --candidate artifacts/optimized --tag control --description 'Unchanged baseline versus itself'
```

Initialization freezes the protocol, evaluator sources, reference asset contents,
source weight identities, baseline program hashes, SDK runtime libraries and
FastRPC library. It copies the current executable and adjacent DLLs into the
campaign so rebuilding the native runner cannot change the baseline executable.
Python versions and the Windows power scheme must remain unchanged. The compiled
model files are hashed and referenced in place, not duplicated.

The control should normally be **inconclusive**, with identical output and no
reliable speed gain. A crash, changed output or apparent control win means the
measurement setup needs investigation before optimization. Review the control's
absolute ONNX failures and baseline timing drift.

## Editable scope

Experiment with exporters and semantics-preserving transformations, graph
grouping, invariant computation, shape buckets, compiler options, or
`native/runner.cpp`. Supporting builds and regression tests may change. Prefer
one hypothesis per candidate and explain the expected mechanism and affected
stage before running it.

Preserve the native timing boundaries, clocks and reported units when editing
C++. Every timed request must perform synthesis: do not memoize whole-request
audio/latents or exploit the repeated benchmark inputs. Reusing invariant
conditioning within synthesis is allowed when it preserves the requested work.

Do not edit the evaluator, corpus, protocol, reference assets, upstream
submodules, or preserved baseline during a campaign. The runner checks their
hashes before and after trials and fails closed. Python client/evaluator changes
require a reviewed new campaign. Runtime C++ changes are supported through
separate baseline and candidate executables; the default candidate executable is
the current `build/runner/Release/supertonic_qnn.exe`.

Use profiles to form hypotheses. Keep detailed profiling separate from timing:
only uninstrumented, fully delegated HTP programs qualify. Do not run another
compiler, profiler or NPU workload while measuring. Connect AC power, keep the
same power plan, and allow thermal conditions to settle. The device lock prevents
overlapping runs of this runner in the same checkout; it cannot stop unrelated
processes or another worktree. Coordinate all O16W workloads explicitly.

## Candidate loop

1. Inspect previous results and choose one hypothesis. Do not optimize for a
   particular text or seed or remove a difficult case.
2. Implement in the experiment worktree, run relevant tests, and commit the
   candidate there. Keep the last accepted commit identifiable.
3. Build into a new artifact directory. Reuse unchanged components. For an
   optional bounded build, create a JSON file containing the executable and its
   argument strings, and pass it with `--build-command`. This is an argv array,
   not a PowerShell command string. Build and measurement have separate budgets.
4. Run the complete frozen evaluation:

   ```powershell
   python research/run.py run --campaign artifacts/research/oct16 --candidate artifacts/my-candidate --tag trial01 --description 'Hypothesis and change'
   ```

5. Inspect `result.json`, the TSV ledger, logs and paired WAVs. `keep` means the
   candidate passed this execution/latency screen; it does not certify perceptual
   quality or automatically promote the model.
6. Retain a winner on the experiment branch. For a rejected candidate, revert
   only that experiment's commit or continue from the last accepted commit in
   the isolated worktree. Preserve the result directory and rejected commit for
   reproducibility. Never run the upstream guide's reset loop in the user's
   working checkout.
7. Repeat within the agreed budget. The campaign baseline stays fixed. To make
   an accepted candidate the new comparator, initialize a new campaign and run
   its control; do not edit an existing campaign manifest.

A unique tag is required for every trial; existing result directories are never
overwritten. All generated results live under ignored `artifacts/`. Preserve a
concise evidence snapshot in `benchmarks/` when reporting a completed batch.

## Measurement and decisions

The default protocol uses seven cases across five languages, two fixed seeds,
eight flow steps and speed 1.05. Each candidate first gets ONNX quality checks
with three requests per case/seed. Only then does it get three separate timing
rounds with 16 requests per case/seed, discarding the first request each time.
Variant order alternates across cases and reverses in the next timing round.
Timing-only rounds do not run ONNX comparisons. Each phase logs raw timings.

The score is the geometric mean of candidate/baseline **median native synthesis
latency ratios**, equally weighted across cases and seeds. Lower is better.
Report RTF and p90 alongside it. The measured interval excludes loading, Python
preprocessing and file I/O; this protocol makes no cold-start, streaming
time-to-first-audio or application end-to-end latency claim.

Acceptance requires all of the following:

- Identical final valid latents and audio, unchanged duration and noise capacity,
  matching source weights, FP16/exact activations, and no CPU compute fallback.
- No new or worsened absolute ONNX failures. Existing baseline failures remain
  explicitly listed as **partial** absolute qualification. They are never
  relabeled as passes just because the candidate matches the baseline.
- More than 3% aggregate latency reduction in **every** timing round, with the
  overall reduction larger than the largest baseline timing drift across rounds
  for any case/seed. Drift is `max(baseline median) / min(baseline median) - 1`.
- No case/seed median regression above 5%, or p90 regression above 10%, in any
  round. A repeatable violation is a discard; a transient violation is
  inconclusive.
- Total referenced program bytes at most 1.25 times the baseline. Shared paths
  are counted once. Peak resident memory is **not measured** by this initial
  harness; the result records null. Add a fixed memory measurement protocol
  before making memory-efficiency claims or pursuing large memory tradeoffs.

This is a conservative repeatability heuristic, not a confidence interval or
proof of statistical significance. The README records 22.9% drift for an
unchanged model; tiny single-run improvements are not evidence. `inconclusive`
is an expected and useful result. Do not rerun until one trial happens to win;
report all trials and investigate environmental drift first.

The default timeouts are 30 minutes for an optional build, 15 minutes per seed's
quality batch, and five minutes per seed's timing batch. These accommodate QNN
compilation costs; the upstream five-minute training budget is inapplicable.
Timeouts and interrupts terminate the child process tree and record a crash.
After a forced termination of the orchestrator, inspect any remaining processes
before manually removing a stale `artifacts/research/device.lock`.

Result statuses are `keep`, `discard`, `inconclusive`, and `crash`. The CLI exits
zero for completed decisions, including discard/inconclusive, and nonzero for
execution or contract failures. Read the status, not just the process exit code.
The runner never commits, resets, promotes artifacts, or starts another trial
automatically.
