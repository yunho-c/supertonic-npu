"""Apply the small, reproducible X2 Elite target addition to ExecuTorch."""

from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / ".deps" / "executorch"


def replace_once(path: Path, before: str, after: str) -> None:
    text = path.read_text(encoding="utf-8")
    if after in text:
        return
    if text.count(before) != 1:
        raise RuntimeError(f"Unexpected upstream content in {path}")
    path.write_text(text.replace(before, after), encoding="utf-8")


def main() -> None:
    for name in ("program.fbs", "scalar_type.fbs"):
        shutil.copyfile(SOURCE / "schema" / name, SOURCE / "exir/_serialize" / name)
    schema = SOURCE / "backends/qualcomm/serialization"
    replace_once(
        schema / "qc_schema.py",
        "    SC8380XP = 60  # v73\n",
        "    SC8380XP = 60  # v73\n    SC8480XP = 88  # v81, Snapdragon X2 Elite\n",
    )
    replace_once(
        schema / "qc_schema.py",
        "    QcomChipset.SC8380XP: SocInfo(QcomChipset.SC8380XP, HtpInfo(HtpArch.V73, 8)),\n",
        "    QcomChipset.SC8380XP: SocInfo(QcomChipset.SC8380XP, HtpInfo(HtpArch.V73, 8)),\n"
        "    QcomChipset.SC8480XP: SocInfo(QcomChipset.SC8480XP, HtpInfo(HtpArch.V81, 8)),\n",
    )
    replace_once(
        schema / "qc_compiler_spec.fbs",
        "  SC8380XP = 60,\n",
        "  SC8380XP = 60,\n  SC8480XP = 88,\n",
    )
    print("Prepared SC8480XP (SoC 88, HTP V81, requested VTCM 8 MB)")


if __name__ == "__main__":
    main()
