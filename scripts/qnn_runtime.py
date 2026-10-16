"""Shared Python 3.14 client for the native ARM64 inference process."""

import json
import os
from pathlib import Path
import subprocess

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SDK = Path(r"C:\Qualcomm\AIStack\QAIRT\2.50.40.260831")


def tensor_file(directory: Path, name: str, value: np.ndarray) -> dict:
    directory.mkdir(parents=True, exist_ok=True)
    value = np.ascontiguousarray(value)
    path = directory / f"{name}.raw"
    value.tofile(path)
    return {"path": str(path.resolve()), "shape": list(value.shape), "dtype": str(value.dtype)}


def metrics(actual: np.ndarray, expected: np.ndarray) -> dict:
    if actual.shape != expected.shape or not np.isfinite(actual).all():
        raise ValueError("Output shape mismatch or non-finite values")
    actual, expected = actual.astype(np.float64).ravel(), expected.astype(np.float64).ravel()
    error = actual - expected
    signal = float(np.dot(expected, expected))
    noise = float(np.dot(error, error))
    return {
        "max_error": float(np.abs(error).max()),
        "rmse": float(np.sqrt(np.mean(error * error))),
        "relative_l2": float(np.sqrt(noise / max(signal, 1e-30))),
        "cosine": float(np.dot(actual, expected) / max(np.linalg.norm(actual) * np.linalg.norm(expected), 1e-30)),
        "sqnr_db": float(10 * np.log10(max(signal, 1e-30) / max(noise, 1e-30))),
    }


class NativeRunner:
    def __init__(self, models: Path, log: Path, sdk: Path = DEFAULT_SDK):
        env = os.environ.copy()
        drivers = list(Path(r"C:\Windows\System32\DriverStore\FileRepository").glob("qcnspmcdm8480.inf_arm64_*/libcdsprpc.dll"))
        if not drivers:
            raise FileNotFoundError("Snapdragon X2 Elite FastRPC driver was not found")
        env["PATH"] = os.pathsep.join([str(sdk / "lib/aarch64-windows-msvc"), str(drivers[0].parent), env["PATH"]])
        env["ADSP_LIBRARY_PATH"] = str(sdk / "lib/hexagon-v81/unsigned")
        log.parent.mkdir(parents=True, exist_ok=True)
        self.log = log.open("w", encoding="utf-8")
        self.process = subprocess.Popen(
            [str(ROOT / "build/runner/Release/supertonic_qnn.exe"), str(models.resolve())],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.log,
            text=True, encoding="utf-8", env=env,
        )

    def request(self, request: dict) -> dict:
        self.process.stdin.write(json.dumps(request) + "\n")
        self.process.stdin.flush()
        while line := self.process.stdout.readline():
            try:
                response = json.loads(line)
            except json.JSONDecodeError:
                self.log.write(line)
                self.log.flush()
                continue
            if not response.get("ok"):
                raise RuntimeError(response)
            return response["timings"]
        raise RuntimeError(f"Native runner exited with code {self.process.wait()}; see {self.log.name}")

    def close(self):
        self.process.stdin.close()
        try:
            self.process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            self.process.terminate()
            self.process.wait(timeout=10)
        self.log.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
