"""Repeat the frozen Superfast timing after an explicitly recorded environment change.

Preserves the original campaign and reuses its exact timing implementation.
Requires the same exclusive resource lease as measure_superfast.py.
"""
import argparse
import math
from pathlib import Path
import shutil
import statistics

import measure_superfast as c


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=c.ROOT / "artifacts/superfast-oct18-rdp-off")
    parser.add_argument("--reason", default="User requested a new measurement after disconnecting RDP")
    args = parser.parse_args()
    original = c.OUT
    c.verify()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    protocol = c.read(original / "protocol.json")
    protocol.update(created_utc=c.now(), reason=args.reason, previous_campaign=str(original),
                    reruns="One explicitly requested new batch; original retained. No within-batch retries or best-of selection.")
    c.save(output / "protocol.json", protocol)
    manifest = c.read(original / "manifest.json")
    identities = dict(manifest["identities"])
    preserved = (original / "manifest.json", original / "summary.json", original / "timing.json",
                 original / "quality.json", original / "asr.json", original / "artifact-hashes.json")
    for path in (*preserved, output / "protocol.json", Path(__file__).resolve()):
        identities[str(path)] = c.sha(path)
    for case in protocol["timing_cases"]:
        for role in ("baseline", "candidate"):
            for filename in ("audio.raw", "latent.raw", "duration.raw", "text_emb.raw", "supertonic_htp.wav"):
                path = original / "quality" / f"{case['id']}-{case['seed']}" / role / filename
                identities[str(path)] = c.sha(path)
    c.save(output / "manifest.json", {**manifest, "created_utc": c.now(), "reason": args.reason, "identities": identities,
                                      "original_manifest_sha256": c.sha(original / "manifest.json")})
    sources = output / "sources"
    sources.mkdir()
    for path in (Path(__file__).resolve(), Path(c.__file__).resolve()):
        shutil.copyfile(path, sources / path.name)
    c.OUT = output
    c.timing()
    timing = c.read(output / "timing.json")
    assert timing["complete"]
    rows = timing["rows"]
    # A timing-only repeat may reuse quality evidence only for bit-identical outputs.
    checks = []
    for case in protocol["timing_cases"]:
        for r in range(3):
            for label, role in (("baseline-before", "baseline"), ("candidate", "candidate"), ("baseline-after", "baseline")):
                directory = output / "timing" / f"round-{r}" / case["id"] / label
                reference = original / "quality" / f"{case['id']}-{case['seed']}" / role
                for name in ("audio.raw", "latent.raw", "duration.raw", "text_emb.raw", "supertonic_htp.wav"):
                    actual, expected = c.sha(directory / name), c.sha(reference / name)
                    assert actual == expected, directory / name
                    checks.append({"path": str(directory / name), "sha256": actual, "reference": str(reference / name)})
    gm = lambda values: math.exp(statistics.mean(math.log(x) for x in values))
    gain = 100 * (1 - gm(r["ratio"] for r in rows))
    round_gains = [100 * (1 - gm(r["ratio"] for r in rows if r["round"] == i)) for i in range(3)]
    local = 100 * max(r["bracket_drift"] for r in rows)
    drift, cases = local, {}
    for name in sorted({r["case"] for r in rows}):
        subset = [r for r in rows if r["case"] == name]
        inter = 100 * (max(r["baseline_ms"] for r in subset) / min(r["baseline_ms"] for r in subset) - 1)
        drift = max(drift, inter)
        cases[name] = {key: statistics.median(r[key] for r in subset) for key in
                       ("baseline_ms", "candidate_ms", "candidate_p90_ms", "baseline_p90_ms", "candidate_request_ms",
                        "baseline_request_ms", "audio_seconds", "baseline_audio_seconds", "inference_rtf", "baseline_duration_rtf", "request_rtf")}
        cases[name].update(gain_percent=100 * (1 - gm(r["ratio"] for r in subset)), interround_baseline_drift_percent=inter,
                           stage_ms={s: {role: statistics.median(r[role + "_stage_ms"][s] for r in subset)
                                        for role in ("baseline", "candidate")} for s in c.STAGES})
    summary = {"completed_utc": c.now(), "reason": args.reason, "scope": protocol, "cases": cases, "gain_percent": gain,
               "round_gains_percent": round_gains, "max_local_bracket_drift_percent": local, "max_drift_percent": drift,
               "repeatable_gain": all(g > 0 for g in round_gains) and gain > drift,
               "target_rtf_met_every_case_round": all(r["inference_rtf"] <= .01 and r["baseline_duration_rtf"] <= .01 for r in rows),
               "timing_outputs_bit_exact_to_previous_quality": True,
               "previous_quality": str(original / "quality.json"), "previous_asr": str(original / "asr.json"),
               "quality_note": "Same compiled programs; EN/KO timing outputs match prior quality byte-for-byte. Six-case ASR was not rerun.",
               "causality": "RDP disconnection was user-reported. This before/after comparison does not isolate RDP from every other system change.",
               "source_manifest_sha256": c.sha(output / "manifest.json")}
    c.save(output / "summary.json", summary)
    c.save(output / "output-parity.json", checks)
    c.save(c.ROOT / "benchmarks" / f"{output.name}.json", {**summary, "timing_rounds": rows, "control": c.read(output / "control.json")})
    commands = [c.read(p) for p in output.rglob("command.json")]
    assert len(commands) == 24 and all(x.get("exit_code") == 0 and x.get("completed_utc") for x in commands)
    c.verify()
    hashes = {str(p.relative_to(output)): c.sha(p) for p in sorted(output.rglob("*")) if p.is_file()}
    c.save(output / "artifact-hashes.json", hashes)
    c.save(output / "audit.json", {"completed_utc": c.now(), "all_frozen_identities_verified": True,
                                    "output_parity_checks": len(checks), "terminal_native_commands": len(commands),
                                    "warm_requests": 360, "artifact_count": len(hashes)})
    print("REPEAT COMPLETE", {"gain_percent": gain, "round_gains": round_gains, "max_drift_percent": drift, "cases": cases}, flush=True)


if __name__ == "__main__":
    main()
