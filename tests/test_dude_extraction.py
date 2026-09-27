"""Archive access must stay inside the explicit training-asset allowlist."""
import importlib.util
from pathlib import Path
import pytest

SPEC = importlib.util.spec_from_file_location('dude_extraction', Path(__file__).resolve().parents[1] / 'experiments/extract_dude_sources.py')
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)
DOC = 'a' * 32

def test_only_requested_training_and_azure_assets_are_selected():
    assert module.selected_asset(f'package/PDF/train/{DOC}.pdf', {DOC}) == (DOC, 'pdf', f'pdf/{DOC}.pdf')
    assert module.selected_asset(f'package/OCR/Azure/{DOC}_original.json', {DOC}) == (
        DOC, 'azure_original', f'azure/{DOC}_original.json')
    for name in (f'package/PDF/test/{DOC}.pdf', f'package/PDF/val/{DOC}.pdf',
                 f'package/OCR/Amazon/{DOC}_due.json', 'package/README.py'):
        assert module.selected_asset(name, {DOC}) is None
    assert module.selected_asset(f'package/PDF/train/{DOC}.pdf', set()) is None

@pytest.mark.parametrize('name', ['../PDF/train/a.pdf', '/PDF/train/a.pdf',
    'x/../../outside', 'C:/outside', 'package\\PDF\\train\\a.pdf'])
def test_unsafe_archive_paths_are_rejected_even_if_not_allowlisted(name):
    with pytest.raises(ValueError, match='Unsafe'):
        module.selected_asset(name, {DOC})

def test_destination_cannot_escape_root(tmp_path):
    assert module.destination(tmp_path, f'pdf/{DOC}.pdf').is_relative_to(tmp_path)
    with pytest.raises(ValueError, match='leaves'):
        module.destination(tmp_path, '../outside.pdf')


def test_bad_source_size_is_explicit_exclusion_without_relaxing_bound():
    assert module.asset_size_exclusion(0) == 'empty_source_asset'
    assert module.asset_size_exclusion(1) is None
    assert module.asset_size_exclusion(module.MAX_ASSET_BYTES) is None
    assert module.asset_size_exclusion(module.MAX_ASSET_BYTES + 1) == 'source_asset_exceeds_fixed_512MiB_limit'
    for invalid in (-1, True, 1.5):
        with pytest.raises(ValueError, match='Invalid'):
            module.asset_size_exclusion(invalid)
