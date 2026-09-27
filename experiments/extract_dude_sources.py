"""Extract allowlisted training PDFs and Azure OCR without executing archive code."""
from __future__ import annotations
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import tarfile
import time

ARCHIVE_SIZE = 21240289931
ARCHIVE_SHA256 = '1506384a93022a2da6b180270345a6928ea7347842eaa2d2e177190c4fd29cae'
CENSUS_SHA256 = '56e6927115a0f85638ae427e29fd91b5d0ee7ee0997c38e842b063a6bda26a58'
MAX_ASSET_BYTES = 512 << 20

def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()

def selected_asset(member_name: str, allowed: set[str]):
    path = PurePosixPath(member_name)
    if path.is_absolute() or '..' in path.parts or '\\' in member_name or ':' in member_name:
        raise ValueError('Unsafe archive member name')
    name = path.as_posix()
    pdf = re.search(r'(?:^|/)PDF/train/([0-9a-f]{32})\.pdf$', name)
    ocr = re.search(r'(?:^|/)OCR/Azure/([0-9a-f]{32})_(due|original)\.json$', name)
    if pdf and pdf[1] in allowed:
        return pdf[1], 'pdf', f'pdf/{pdf[1]}.pdf'
    if ocr and ocr[1] in allowed:
        return ocr[1], 'azure_' + ocr[2], f'azure/{ocr[1]}_{ocr[2]}.json'
    return None

def destination(root: Path, relative: str):
    root = root.resolve()
    target = (root / relative).resolve()
    if not target.is_relative_to(root):
        raise ValueError('Asset destination leaves requested root')
    return target


def asset_size_exclusion(size: int):
    """Bad source assets are logged as unavailable, never silently repaired."""
    if type(size) is not int or size < 0:
        raise ValueError('Invalid archive asset size')
    if size == 0:
        return 'empty_source_asset'
    if size > MAX_ASSET_BYTES:
        return 'source_asset_exceeds_fixed_512MiB_limit'
    return None

def extract(archive: Path, census: Path, output: Path):
    archive, census, output = [x.resolve() for x in (archive, census, output)]
    if archive.stat().st_size != ARCHIVE_SIZE or sha256_file(archive) != ARCHIVE_SHA256:
        raise ValueError('Archive differs from pinned creator release')
    if sha256_file(census) != CENSUS_SHA256:
        raise ValueError('Candidate census differs from the audited planning release')
    data = json.loads(census.read_text(encoding='utf-8'))
    allowed = {row['doc_id'] for row in data['candidate_question_geometry']}
    assert len(allowed) == 2897 and all(re.fullmatch('[0-9a-f]{32}', x) for x in allowed)
    output.mkdir(parents=True, exist_ok=True)
    started = last = time.monotonic()
    assets, seen, stats, excluded_assets = [], set(), Counter(), []
    journal = output / 'extracted_assets.jsonl'
    # Each restart verifies existing individual assets against the archive bytes.
    with journal.open('w', encoding='utf-8', newline='\n') as log:
        with tarfile.open(archive, mode='r|gz', bufsize=8 << 20) as stream:
            for member in stream:
                stats['members_scanned'] += 1
                if time.monotonic() - last > 30:
                    print(f'Scanned {stats["members_scanned"]} archive members; {len(assets)} selected assets verified', flush=True)
                    last = time.monotonic()
                selected = selected_asset(member.name, allowed)
                if selected is None:
                    continue
                if not member.isfile() or member.issym() or member.islnk():
                    raise ValueError('Selected archive member is not a regular file')
                doc_id, kind, relative = selected
                if relative in seen:
                    raise ValueError('Duplicate selected destination in archive')
                seen.add(relative)
                exclusion = asset_size_exclusion(member.size)
                if exclusion:
                    record = {'doc_id': doc_id, 'kind': kind, 'archive_member': member.name,
                              'bytes': member.size, 'reason': exclusion}
                    excluded_assets.append(record)
                    stats['excluded_' + exclusion] += 1
                    print('Source asset unavailable: ' + json.dumps(record), flush=True)
                    continue
                target = destination(output, relative)
                target.parent.mkdir(parents=True, exist_ok=True)
                digest = hashlib.sha256()
                content = stream.extractfile(member)
                assert content is not None
                partial = target.with_name(target.name + '.part')
                if partial.exists():
                    # This is a verified fixed path within the requested extraction root.
                    partial.unlink()
                with partial.open('xb') as dest:
                    copied = 0
                    while chunk := content.read(8 << 20):
                        copied += len(chunk)
                        if copied > member.size:
                            raise ValueError('Asset stream exceeds declared member size')
                        dest.write(chunk)
                        digest.update(chunk)
                if copied != member.size:
                    raise ValueError('Truncated selected asset')
                asset_sha = digest.hexdigest()
                if target.exists():
                    if target.stat().st_size != member.size or sha256_file(target) != asset_sha:
                        raise ValueError('Existing asset differs; refusing overwrite')
                    partial.unlink()
                else:
                    partial.replace(target)
                row = {'doc_id': doc_id, 'kind': kind, 'relative_path': relative,
                       'archive_member': member.name, 'bytes': copied, 'sha256': asset_sha}
                assets.append(row)
                log.write(json.dumps(row, ensure_ascii=False) + '\n')
                log.flush()
                stats[kind + '_files'] += 1
                stats[kind + '_bytes'] += copied
                if time.monotonic() - last > 15:
                    print(f'Extracted {len(assets)} assets; {stats["members_scanned"]} archive members scanned', flush=True)
                    last = time.monotonic()
    per_doc = {doc: set() for doc in allowed}
    for row in assets:
        per_doc[row['doc_id']].add(row['kind'])
    missing = {doc: sorted({'pdf', 'azure_due', 'azure_original'} - kinds)
               for doc, kinds in sorted(per_doc.items()) if len(kinds) != 3}
    result = {'status': 'source_assets_extracted_without_model_inference',
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'archive_sha256': ARCHIVE_SHA256, 'archive_bytes': ARCHIVE_SIZE,
        'author_revision': 'b3662175d3b2482d711f18559b7acc2a5bccc600',
        'archive_fully_scanned': True, 'verified_complete': True,
        'census_sha256': CENSUS_SHA256, 'script_sha256': sha256_file(Path(__file__)),
        'allowed_training_doc_ids': len(allowed), 'complete_document_asset_sets': len(allowed)-len(missing),
        'missing_assets_by_doc_id': missing, 'statistics': dict(stats),
        'source_asset_exclusions': excluded_assets,
        'asset_manifest_sha256': sha256_file(journal), 'duration_seconds': time.monotonic()-started,
        'source_code_from_archive_executed': False, 'validation_or_test_pdfs_extracted': False}
    (output / 'extraction.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({k: v for k, v in result.items() if k != 'missing_assets_by_doc_id'}, indent=2), flush=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--census', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    extract(args.archive, args.census, args.output)
