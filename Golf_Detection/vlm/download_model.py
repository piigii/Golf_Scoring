"""Download the requested checkpoint into vlm/models with a pinned revision."""
import argparse
import json
import os
import hashlib
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("HF_HOME", str(ROOT / ".cache/huggingface"))
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")


def digest(path):
    sha = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            sha.update(block)
    return sha.hexdigest()


def download_weight(repo, revision, file, target):
    """Bounded parallel HTTP ranges, resumable chunks, full SHA-256 verification."""
    size, expected = file.lfs.size, file.lfs.sha256
    output = target / file.rfilename
    if output.is_file() and output.stat().st_size == size and digest(output) == expected:
        print(f"Verified existing {file.rfilename}", flush=True)
        return expected
    cache = ROOT / ".cache/ranges" / expected
    cache.mkdir(parents=True, exist_ok=True)
    chunk_size = 32 * 1024 * 1024
    ranges = [(i, start, min(start + chunk_size, size) - 1)
              for i, start in enumerate(range(0, size, chunk_size))]

    def fetch(item):
        i, start, end = item
        part = cache / f"{i:05d}.part"
        if part.is_file() and part.stat().st_size == end - start + 1:
            return part
        temporary = part.with_suffix(".tmp")
        url = f"https://huggingface.co/{repo}/resolve/{revision}/{file.rfilename}?offset={start}"
        for attempt in range(4):
            try:
                request = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}"})
                with urllib.request.urlopen(request, timeout=60) as response, temporary.open("wb") as out:
                    if response.status != 206 or response.headers.get("Content-Range") != f"bytes {start}-{end}/{size}":
                        raise RuntimeError("Server did not return the exact requested byte range")
                    while block := response.read(1024 * 1024):
                        out.write(block)
                if temporary.stat().st_size != end - start + 1:
                    raise RuntimeError("Incomplete HTTP range")
                temporary.replace(part)
                return part
            except Exception:
                if attempt == 3:
                    raise
                time.sleep(2 ** attempt)

    with ThreadPoolExecutor(max_workers=8) as pool:
        for done, future in enumerate(as_completed([pool.submit(fetch, item) for item in ranges]), 1):
            future.result()
            if done % 4 == 0 or done == len(ranges):
                print(f"{file.rfilename}: {done}/{len(ranges)} chunks", flush=True)
    temporary = output.with_suffix(".assembling")
    sha = hashlib.sha256()
    with temporary.open("wb") as out:
        for i, _, _ in ranges:
            with (cache / f"{i:05d}.part").open("rb") as source:
                while block := source.read(1024 * 1024):
                    sha.update(block)
                    out.write(block)
    if sha.hexdigest() != expected:
        raise RuntimeError(f"SHA-256 mismatch: {file.rfilename}; no checkpoint was installed")
    temporary.replace(output)
    for part in cache.glob("*.part"):
        part.unlink()
    print(f"SHA-256 verified: {file.rfilename}", flush=True)
    return expected


def main():
    from huggingface_hub import HfApi, snapshot_download
    from backend import CUDA_REVISION, default_model, select_backend
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=["auto", "metal", "cuda"], default="auto")
    parser.add_argument("--revision", help="Override the pinned Hugging Face commit")
    args = parser.parse_args()
    backend = select_backend(args.backend)
    repo = "eadx/LocateAnything-3B-MLX" if backend == "metal" else "nvidia/LocateAnything-3B"
    revision = args.revision or ("010e23e085e4a3ce41ba1666b0c6ce356b737e14" if backend == "metal" else CUDA_REVISION)
    info = HfApi().model_info(repo, revision=revision, files_metadata=True)
    target = default_model(backend)
    print(f"Downloading {repo}@{info.sha} to {target}", flush=True)
    snapshot_download(repo_id=repo, revision=info.sha, local_dir=target, max_workers=4,
                      ignore_patterns=["*.safetensors"])
    hashes = {}
    for file in info.siblings:
        if file.rfilename.endswith(".safetensors"):
            hashes[file.rfilename] = download_weight(repo, info.sha, file, target)
    manifest = {"repo_id": repo, "revision": info.sha,
                "model_path": str(target.relative_to(ROOT)), "sha256": hashes,
                "files": [{"path": str(p.relative_to(target)), "bytes": p.stat().st_size}
                          for p in sorted(target.rglob("*")) if p.is_file() and ".cache" not in p.parts and p.name != "download_manifest.json"]}
    (target / "download_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print("Download completed.", flush=True)


if __name__ == "__main__":
    main()
