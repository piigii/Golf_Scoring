"""Package a full workspace into checksum-verified GitHub Release-sized parts."""
import argparse
import hashlib
import io
import json
import os
import subprocess
import tarfile
from pathlib import Path

EXCLUDED = {'.git', '.aws', '.codex', '.agents', '.DS_Store', '.env', 'token', 'stored_tokens'}


class PartsWriter(io.RawIOBase):
    def __init__(self, directory, limit):
        self.directory, self.limit = directory, limit
        self.parts, self.stream, self.size, self.total = [], None, 0, 0

    def writable(self):
        return True

    def tell(self):
        return self.total

    def write(self, data):
        original = len(data)
        data = memoryview(data)
        while data:
            if self.stream is None:
                self.name = f'workspace.tar.gz.part{len(self.parts) + 1:03d}'
                self.stream = (self.directory / self.name).open('wb')
                self.digest, self.size = hashlib.sha256(), 0
            n = min(len(data), self.limit - self.size)
            block, data = data[:n], data[n:]
            self.stream.write(block)
            self.digest.update(block)
            self.size += n
            self.total += n
            if self.size == self.limit:
                self.finish_part()
        return original

    def finish_part(self):
        if self.stream is not None:
            self.stream.close()
            self.parts.append(dict(name=self.name, bytes=self.size, sha256=self.digest.hexdigest()))
            print(f'Packed {self.name}: {self.size / 1024**3:.2f} GiB', flush=True)
            self.stream = None


def package(root, output, limit=1024**3, runtime=None):
    root, output = root.resolve(), output.resolve()
    runtime = runtime.resolve() if runtime else None
    if output == root or root in output.parents:
        raise ValueError('Put backup files outside the source workspace')
    output.mkdir(parents=True, exist_ok=True)
    if list(output.glob('workspace.tar.gz.part*')):
        raise FileExistsError('Output already contains archive parts')
    writer = PartsWriter(output, limit)
    count, input_bytes, partial = 0, 0, []

    def select(info):
        nonlocal count, input_bytes
        components = Path(info.name).parts
        if any(c in EXCLUDED or (c.startswith('.env.') and c != '.env.example') for c in components):
            return None
        if info.issym() and os.path.isabs(info.linkname):
            resolved = Path(info.linkname).resolve()
            if runtime and (resolved == runtime or runtime in resolved.parents):
                target = Path('_runtime/python') / resolved.relative_to(runtime)
                info.linkname = os.path.relpath(target, Path(info.name).parent)
            elif resolved == root or root in resolved.parents:
                info.linkname = os.path.relpath(resolved.relative_to(root), Path(info.name).parent)
            else:
                raise ValueError(f'Cannot archive external absolute symlink: {info.name}')
        if info.isfile():
            count += 1
            input_bytes += info.size
            if info.name.endswith('.crdownload') or (info.size == 0 and info.name.endswith('.zip')):
                partial.append(dict(path=info.name, bytes=info.size, status='incomplete download or empty archive'))
        return info

    with tarfile.open(fileobj=writer, mode='w|gz', compresslevel=1) as archive:
        for child in sorted(root.iterdir()):
            archive.add(child, arcname=child.name, filter=select)
        if runtime:
            archive.add(runtime, arcname='_runtime/python', filter=select)
    writer.finish_part()
    manifest = dict(format='split-tar-gzip-v1', parts=writer.parts,
                    compressed_bytes=writer.total, input_file_bytes=input_bytes, file_count=count,
                    excluded_names=sorted(EXCLUDED), incomplete_inputs=partial,
                    python_runtime='_runtime/python' if runtime else None,
                    environment_platform='macOS arm64; recreate environments on CUDA/Linux',
                    source_commit=subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip())
    (output / 'workspace-manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    (output / 'SHA256SUMS').write_text(''.join(f"{p['sha256']}  {p['name']}\n" for p in writer.parts))
    print(json.dumps({k: manifest[k] for k in ('file_count', 'input_file_bytes', 'compressed_bytes')}, indent=2), flush=True)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--runtime', type=Path, help='Optional standalone Python installation to bundle')
    args = parser.parse_args()
    package(args.root, args.output, runtime=args.runtime.resolve() if args.runtime else None)


if __name__ == '__main__':
    main()
