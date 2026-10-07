"""Download, verify and restore the full workspace snapshot from GitHub Releases."""
import argparse
import hashlib
import io
import json
import os
import platform
import shutil
import tarfile
import time
import urllib.request
from pathlib import Path

DEFAULT_RELEASE = 'workspace-full-2026-10-07'
BASE_URL = 'https://github.com/piigii/Golf_Scoring/releases/download'


class JoinedReader(io.RawIOBase):
    def __init__(self, files):
        self.files = iter(files)
        self.stream = None

    def readable(self):
        return True

    def read(self, size=-1):
        if size < 0:
            raise ValueError('Use streaming reads for large archives')
        chunks = []
        remaining = size
        while remaining:
            if self.stream is None:
                try:
                    self.stream = next(self.files).open('rb')
                except StopIteration:
                    break
            block = self.stream.read(remaining)
            if block:
                chunks.append(block)
                remaining -= len(block)
            else:
                self.stream.close()
                self.stream = None
        return b''.join(chunks)

    def close(self):
        if self.stream:
            self.stream.close()
        super().close()


def verify(path, part):
    if not path.is_file() or path.stat().st_size != part['bytes']:
        return False
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024**2), b''):
            digest.update(block)
    return digest.hexdigest() == part['sha256']


def download(url, path):
    temp = path.with_name(path.name + '.download')
    for attempt in range(4):
        try:
            request = urllib.request.Request(url, headers={'User-Agent': 'Golf-Scoring-restore'})
            with urllib.request.urlopen(request, timeout=120) as source, temp.open('wb') as target:
                shutil.copyfileobj(source, target, 4 * 1024**2)
            temp.replace(path)
            return
        except Exception:
            if attempt == 3:
                raise
            time.sleep(2**attempt)


def extract(parts, destination):
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError('Choose an empty destination; existing work will not be overwritten')
    destination.mkdir(parents=True, exist_ok=True)
    with JoinedReader(parts) as stream, tarfile.open(fileobj=stream, mode='r|gz') as archive:
        archive.extractall(destination, filter='data')


def relocate_environments(destination):
    if platform.system() != 'Darwin' or platform.machine() != 'arm64':
        raise RuntimeError('Bundled Python and venvs are macOS ARM64 only; recreate environments on this platform')
    runtime = destination / '_runtime/python/bin'
    if not (runtime / 'python3.11').is_file():
        raise FileNotFoundError('Bundled Python runtime missing')
    for relative in ('Golf_Detection/.venv', 'Golf_Detection/vlm/.venv'):
        venv = destination / relative
        config = venv / 'pyvenv.cfg'
        if not config.exists():
            continue
        lines = config.read_text().splitlines()
        old_bin = next(line.split('=', 1)[1].strip() for line in lines if line.startswith('home ='))
        # Replace paths in activation scripts and entry-point shebangs only.
        old_root = '/Users/piigii/Documents/Golf_Scoring'
        for file in (venv / 'bin').iterdir():
            if file.is_file() and not file.is_symlink() and file.stat().st_size < 1024**2:
                try:
                    text = file.read_text()
                except UnicodeDecodeError:
                    continue
                updated = text.replace(old_root, str(destination)).replace(old_bin, str(runtime))
                if updated != text:
                    file.write_text(updated)
        config.write_text('\n'.join('home = ' + str(runtime) if line.startswith('home =') else line for line in lines) + '\n')
        python = venv / 'bin/python'
        if python.is_symlink():
            python.unlink()
        python.symlink_to(os.path.relpath(runtime / 'python3.11', python.parent))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--release', default=DEFAULT_RELEASE)
    parser.add_argument('--downloads', type=Path, default=Path('workspace-downloads'))
    parser.add_argument('--destination', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, help='Use already downloaded parts beside a local manifest')
    parser.add_argument('--relocate-python', action='store_true', help='Repoint bundled macOS ARM64 Python environments')
    args = parser.parse_args()
    if args.manifest:
        manifest_path = args.manifest
        downloads = manifest_path.parent
    else:
        downloads = args.downloads
        downloads.mkdir(parents=True, exist_ok=True)
        manifest_path = downloads / 'workspace-manifest.json'
        download(f'{BASE_URL}/{args.release}/workspace-manifest.json', manifest_path)
    manifest = json.loads(manifest_path.read_text())
    parts = []
    for part in manifest['parts']:
        name = part['name']
        if Path(name).name != name:
            raise ValueError('Invalid archive part name')
        path = downloads / name
        if not verify(path, part):
            if args.manifest:
                raise ValueError(f'Missing or corrupt part: {name}')
            print(f'Downloading {name} ({part["bytes"]/1024**3:.2f} GiB)', flush=True)
            download(f'{BASE_URL}/{args.release}/{name}', path)
            if not verify(path, part):
                raise ValueError(f'Checksum mismatch: {name}')
        parts.append(path)
    extract(parts, args.destination.resolve())
    if args.relocate_python:
        relocate_environments(args.destination.resolve())
    print(f'Restored full workspace to {args.destination.resolve()}')


if __name__ == '__main__':
    main()
