"""Verify the files listed in the public distribution manifest (standard library)."""
from pathlib import Path, PurePosixPath
import hashlib
import json

ROOT = Path(__file__).resolve().parents[1]


def verify(root=ROOT):
    root = Path(root).resolve()
    manifest = json.loads((root/'release_manifest.json').read_text(encoding='utf-8'))
    seen = set()
    for item in manifest['files']:
        relative = PurePosixPath(item['path'])
        if not item['path'] or relative.as_posix() != item['path']:
            raise ValueError('Noncanonical manifest path')
        if relative.is_absolute() or '..' in relative.parts or '\\' in item['path'] or ':' in item['path']:
            raise ValueError('Invalid manifest path')
        if item['path'] in seen:
            raise ValueError('Duplicate manifest path')
        seen.add(item['path'])
        path = (root/relative).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise ValueError('Missing or out-of-tree file: '+item['path'])
        if hashlib.sha256(path.read_bytes()).hexdigest() != item['sha256']:
            raise ValueError('File digest differs: '+item['path'])
    if not seen:
        raise ValueError('Empty release inventory')
    return len(seen)


if __name__ == '__main__':
    print(f'PASS: {verify()} published file hashes verified.')
