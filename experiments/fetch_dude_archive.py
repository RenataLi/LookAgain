"""Download the pinned public creator-published DUDE PDF/OCR archive."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import time

import requests

REVISION = 'b3662175d3b2482d711f18559b7acc2a5bccc600'
NAME = 'DUDE_train-val-test_binaries.tar.gz'
SIZE = 21240289931
SHA256 = '1506384a93022a2da6b180270345a6928ea7347842eaa2d2e177190c4fd29cae'
URL = f'https://huggingface.co/datasets/jordyvl/DUDE_loader/resolve/{REVISION}/data/{NAME}?download=true'

def file_hash(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()

def download(output: Path):
    output.mkdir(parents=True, exist_ok=True)
    target = output / NAME
    partial = target.with_suffix(target.suffix + '.part')
    if target.exists():
        if target.stat().st_size != SIZE or file_hash(target) != SHA256:
            raise ValueError('Cached archive differs from creator LFS SHA256; refusing overwrite')
        print('Pinned archive already verified', flush=True)
        return
    offset = partial.stat().st_size if partial.exists() else 0
    if offset > SIZE:
        raise ValueError('Partial archive is larger than expected')
    if shutil.disk_usage(output).free < SIZE - offset + (2 << 30):
        raise RuntimeError('Insufficient space for pinned download plus safety margin')
    start = last = time.monotonic()
    if offset < SIZE:
        headers = {'Range': f'bytes={offset}-'} if offset else {}
        with requests.get(URL, headers=headers, stream=True, timeout=(30, 90)) as response:
            response.raise_for_status()
            if offset:
                expected = f'bytes {offset}-'
                if response.status_code != 206 or not response.headers.get('Content-Range', '').startswith(expected):
                    raise RuntimeError('Server did not honour resume range; existing partial preserved')
            elif response.status_code != 200:
                raise RuntimeError('Unexpected status for fresh archive request')
            if 'text/html' in response.headers.get('Content-Type', ''):
                raise RuntimeError('Expected public archive; received an HTML response')
            with partial.open('ab' if offset else 'xb') as stream:
                for chunk in response.iter_content(8 << 20):
                    if not chunk:
                        continue
                    if stream.tell() + len(chunk) > SIZE:
                        raise ValueError('Server exceeded pinned archive size')
                    stream.write(chunk)
                    now = time.monotonic()
                    if now - last > 15:
                        received = stream.tell()
                        print(f'{100*received/SIZE:.1f}% | {received/1e9:.2f}/{SIZE/1e9:.2f} GB | '
                              f'{(received-offset)/(now-start)/1e6:.1f} MB/s', flush=True)
                        last = now
    if partial.stat().st_size != SIZE:
        raise ValueError('Incomplete archive; retained for verified range resume')
    print('Download complete; checking creator SHA256', flush=True)
    if file_hash(partial) != SHA256:
        raise ValueError('Archive checksum differs from creator LFS pointer')
    partial.replace(target)
    provenance = {'source_url': URL, 'revision': REVISION, 'size': SIZE, 'sha256': SHA256,
        'downloaded_utc': datetime.now(timezone.utc).isoformat(),
        'seconds_this_segment': time.monotonic() - start, 'resumed_from_byte': offset,
        'remote_code_executed': False, 'purpose': 'Source-only audited DUDE cohort preparation'}
    (output / (NAME + '.download.json')).write_text(json.dumps(provenance, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(provenance), flush=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    download(parser.parse_args().output)
