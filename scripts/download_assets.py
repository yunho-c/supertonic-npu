"""Download the pinned Supertonic 3 assets without a second cache copy."""

import argparse
import hashlib
import json
from pathlib import Path
from urllib.request import urlopen

REPOSITORY = "supertone-oss-archive/supertonic-3"
REVISION = "aafc6e32416a594460b32413efc49d7fe4ce6d46"


def digest(path: Path, git_blob: bool = False) -> str:
    checksum = hashlib.sha1() if git_blob else hashlib.sha256()
    if git_blob:
        checksum.update(f"blob {path.stat().st_size}\0".encode())
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            checksum.update(block)
    return checksum.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("assets"))
    args = parser.parse_args()
    metadata_url = f"https://huggingface.co/api/models/{REPOSITORY}/revision/{REVISION}?blobs=true"
    with urlopen(metadata_url, timeout=60) as response:
        metadata = json.load(response)
    files = []
    for entry in metadata["siblings"]:
        name = entry["rfilename"]
        if not (name.startswith(("onnx/", "voice_styles/")) or name in ("LICENSE", "config.json")):
            continue
        target = args.output / name
        target.parent.mkdir(parents=True, exist_ok=True)
        lfs = entry.get("lfs")
        expected = lfs["sha256"] if lfs else entry["blobId"]
        if not target.exists() or digest(target, git_blob=not lfs) != expected:
            print(f"Downloading {name} ({entry['size'] / 1e6:.1f} MB)", flush=True)
            partial = target.with_suffix(target.suffix + ".partial")
            url = f"https://huggingface.co/{REPOSITORY}/resolve/{REVISION}/{name}"
            with urlopen(url, timeout=120) as response, partial.open("wb") as output:
                for block in iter(lambda: response.read(1024 * 1024), b""):
                    output.write(block)
            if digest(partial, git_blob=not lfs) != expected:
                raise RuntimeError(f"Checksum mismatch: {name}")
            partial.replace(target)
        files.append({"path": name, "size": target.stat().st_size, "sha256": digest(target)})
    manifest = {"repository": REPOSITORY, "revision": REVISION, "files": files}
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Verified {len(files)} files in {args.output}", flush=True)


if __name__ == "__main__":
    main()
