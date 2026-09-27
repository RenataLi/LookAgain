"""Independent PDF serialization/parser-boundary check; no corpus/model outcomes.

Run from the repository with a Python environment containing pypdf:
python experiments/check_dude_pdf_boundary.py --output reports/dude_preparation/pdf_boundary_check.json
"""
from pathlib import Path
import argparse
import hashlib
import importlib.util
import io
import json
import re
from datetime import datetime, timezone

import pypdf
from pypdf import PdfReader, PdfWriter
from pypdf.generic import NameObject, NumberObject, FloatObject, BooleanObject, TextStringObject, ArrayObject


PROJECT = Path(__file__).resolve().parents[1]


def describe(value):
    return {"type": type(value).__name__, "repr": repr(value),
            "is_int": isinstance(value, int), "is_float": isinstance(value, float)}


def fixture(field=None, value=None, inherited=False, crop=None, indirect=False, parent_rotate=None):
    writer = PdfWriter()
    page = writer.add_blank_page(width=720, height=432)
    target = writer._pages.get_object() if inherited else page
    if parent_rotate is not None:
        writer._pages.get_object()[NameObject('/Rotate')] = parent_rotate
    if field is not None:
        target[NameObject(field)] = writer._add_object(value) if indirect else value
    if crop is not None:
        page[NameObject("/CropBox")] = crop
    buffer = io.BytesIO()
    writer.write(buffer)
    return buffer.getvalue()


def matrix():
    numbers = lambda values: ArrayObject([NumberObject(x) for x in values])
    cases = [
        ("implicit_rotate_0", {}, True),
        ("explicit_number_rotate_0", {"field": "/Rotate", "value": NumberObject(0)}, True),
        ("explicit_float_rotate_0", {"field": "/Rotate", "value": FloatObject(0)}, True),
        ("explicit_number_rotate_90", {"field": "/Rotate", "value": NumberObject(90)}, False),
        ("explicit_float_rotate_point5", {"field": "/Rotate", "value": FloatObject(.5)}, False),
        ("explicit_bool_rotate_false", {"field": "/Rotate", "value": BooleanObject(False)}, False),
        ("explicit_text_rotate_0", {"field": "/Rotate", "value": TextStringObject("0")}, False),
        ("inherited_number_rotate_0", {"field": "/Rotate", "value": NumberObject(0), "inherited": True}, True),
        ("inherited_number_rotate_90", {"field": "/Rotate", "value": NumberObject(90), "inherited": True}, False),
        ("explicit_number_unit_1", {"field": "/UserUnit", "value": NumberObject(1)}, True),
        ("explicit_float_unit_1", {"field": "/UserUnit", "value": FloatObject(1)}, True),
        ("explicit_number_unit_2", {"field": "/UserUnit", "value": NumberObject(2)}, False),
        ("explicit_float_unit_point5", {"field": "/UserUnit", "value": FloatObject(.5)}, False),
        ("explicit_bool_unit_true", {"field": "/UserUnit", "value": BooleanObject(True)}, False),
        ("explicit_text_unit_1", {"field": "/UserUnit", "value": TextStringObject("1")}, False),
        ("number_media_coordinates", {"field": "/MediaBox", "value": numbers([0, 0, 720, 432])}, True),
        ("float_media_coordinates", {"field": "/MediaBox", "value": ArrayObject([FloatObject(x) for x in [0, 0, 720, 432]])}, True),
        ("text_media_width_720", {"field": "/MediaBox", "value": ArrayObject([NumberObject(0), NumberObject(0), TextStringObject("720"), NumberObject(432)])}, False),
        ("bool_media_origin_false", {"field": "/MediaBox", "value": ArrayObject([BooleanObject(False), NumberObject(0), NumberObject(720), NumberObject(432)])}, False),
        ("text_crop_width_720", {"crop": ArrayObject([NumberObject(0), NumberObject(0), TextStringObject("720"), NumberObject(432)])}, False),
        ("bool_crop_origin_false", {"crop": ArrayObject([BooleanObject(False), NumberObject(0), NumberObject(720), NumberObject(432)])}, False),
        ("inherited_number_media_coordinates", {"field": "/MediaBox", "value": numbers([0, 0, 720, 432]), "inherited": True}, True),
        ("inherited_number_crop_coordinates", {"field": "/CropBox", "value": numbers([0, 0, 720, 432]), "inherited": True}, True),
        ("child_zero_overrides_parent_90", {"field": "/Rotate", "value": NumberObject(0), "parent_rotate": NumberObject(90)}, True),
        ("child_90_overrides_parent_zero", {"field": "/Rotate", "value": NumberObject(90), "parent_rotate": NumberObject(0)}, False),
        ("indirect_number_rotate_zero", {"field": "/Rotate", "value": NumberObject(0), "indirect": True}, True),
        ("indirect_number_rotate_90", {"field": "/Rotate", "value": NumberObject(90), "indirect": True}, False),
        ("indirect_number_unit_one", {"field": "/UserUnit", "value": NumberObject(1), "indirect": True}, True),
        ("indirect_text_unit_one", {"field": "/UserUnit", "value": TextStringObject('1'), "indirect": True}, False),
        ("indirect_media_array", {"field": "/MediaBox", "value": numbers([0, 0, 720, 432]), "indirect": True}, True),
        ("indirect_text_media_array", {"field": "/MediaBox", "value": ArrayObject([NumberObject(0), NumberObject(0), TextStringObject('720'), NumberObject(432)]), "indirect": True}, False),
    ]
    # The inherited MediaBox fixture removes the page's own box in run().
    return cases


def pypdf_source_digest():
    """Hash all installed pypdf Python sources in relative-path order."""
    root = Path(pypdf.__file__).resolve().parent
    paths = sorted(root.rglob('*.py'), key=lambda p: p.relative_to(root).as_posix())
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.relative_to(root).as_posix().encode('utf-8'))
        digest.update(b'\0')
        digest.update(path.read_bytes())
        digest.update(b'\0')
    return {'sha256': digest.hexdigest(), 'python_source_file_count': len(paths),
            'algorithm': 'SHA256 over sorted package-relative POSIX paths + NUL + exact .py file bytes + NUL'}


def run():
    path = PROJECT / "experiments/prepare_dude_replication.py"
    spec = importlib.util.spec_from_file_location("dude_boundary_subject", path)
    adapter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(adapter)
    results = []
    for name, kwargs, expected in matrix():
        blob = fixture(**kwargs)
        if name == "inherited_number_media_coordinates":
            reader = PdfReader(io.BytesIO(blob))
            writer = PdfWriter()
            page = writer.add_page(reader.pages[0])
            writer._pages.get_object()[NameObject("/MediaBox")] = page[NameObject("/MediaBox")]
            del page[NameObject("/MediaBox")]
            buffer = io.BytesIO(); writer.write(buffer); blob = buffer.getvalue()
        entry = {"case": name, "expected_geometry_accepted": expected,
                 "pdf_sha256": hashlib.sha256(blob).hexdigest(),
                 "serialized_field_lines": [x.decode('latin1') for x in blob.splitlines()
                     if re.match(rb'/(Rotate|UserUnit|MediaBox|CropBox)\b', x)]}
        expected_error = None if expected else (
            'unsupported_pdf_rotation' if 'rotate' in name or 'overrides' in name else
            'unsupported_pdf_user_unit' if 'unit' in name else 'unsupported_pdf_boxes')
        entry['expected_error'] = expected_error
        validator_reached = False
        try:
            page = PdfReader(io.BytesIO(blob)).pages[0]
            entry["reader_raw"] = {key: describe(page.get(key)) for key in ("/Rotate", "/UserUnit")}
            entry["reader_raw_boxes"] = {key: [describe(v) for v in (page.get(key).get_object() if page.get(key) is not None else [])]
                                         for key in ("/MediaBox", "/CropBox")}
            entry["rotation_property"] = describe(page.rotation)
            pages, _ = adapter._pdf_metadata(blob)
            entry["parsed_geometry"] = {k: describe(v) if k in ("rotation", "user_unit") else v
                                        for k, v in pages[0].items()}
            validator_reached = True
            geometry = adapter.validate_page_geometry(pages[0],
                {"page": 1, "width": 10, "height": 6, "unit": "inch"}, [0, 0, 2000, 1200], 0)
            entry["geometry_accepted"] = True
            entry["render_size"] = geometry["render_size"]
        except Exception as error:
            entry["geometry_accepted"] = False
            entry["error_type"] = type(error).__name__
            entry["error"] = str(error)
        entry['validator_reached'] = validator_reached
        entry["expected_matched"] = (validator_reached and entry["geometry_accepted"] == expected
            and (expected or (entry.get('error_type') == 'Ineligible' and entry.get('error') == expected_error)))
        results.append(entry)
    return {"generated_at": datetime.now(timezone.utc).isoformat(), "pypdf_version": pypdf.__version__,
            'pypdf_source': pypdf_source_digest(),
            'generator': 'experiments/check_dude_pdf_boundary.py',
            'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'adapter': 'experiments/prepare_dude_replication.py',
            "adapter_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "fixture_scope": "synthetic serialized PDFs reopened through PdfReader; no model outcomes",
            "cases": results, "disagreements": [x["case"] for x in results if not x["expected_matched"]]}


def render_review(result, output):
    import os
    link = lambda path: Path(os.path.relpath(path, output.parent)).as_posix()
    lines = ['# Independent PDF parser-boundary check', '',
        f"Generated: {result['generated_at']}", '',
        f"**{sum(x['expected_matched'] for x in result['cases'])}/{len(result['cases'])} engineering cases match their declared outcomes.** These are synthetic PDF checks, not additional scientific examples or model results.", '',
        f"Generator: [check_dude_pdf_boundary.py]({link(PROJECT / result['generator'])}); SHA256 `{result['script_sha256']}`.", '',
        f"Adapter: [prepare_dude_replication.py]({link(PROJECT / result['adapter'])}); SHA256 `{result['adapter_sha256']}`.", '',
        f"pypdf {result['pypdf_version']}; package Python-source SHA256 `{result['pypdf_source']['sha256']}` ({result['pypdf_source']['python_source_file_count']} files). The JSON records the exact digest algorithm.", '',
        f"[Machine-readable evidence]({output.name}) contains actual PDF byte hashes, serialized field lines, reader types and validation errors.", '',
        'The geometry rules remain unchanged: only zero rotation and unit scale are supported. Genuine finite numeric PDF values, including inherited and indirect values, are normalized at the parser boundary. Fractional rotation is never truncated. Numeric-looking strings and Boolean objects are rejected before rectangle or float coercion.', '',
        'For this pypdf version, FloatObject(0.0) rotation reopens as FloatObject(0.0), while FloatObject(1.0) UserUnit serializes as token 1 and reopens as NumberObject(1). The latter is therefore not an independent decimal-token test. Actual serialized tokens and types are retained for every fixture.', '',
        'Inherited MediaBox, CropBox and Rotate, plus child rotation overrides, are checked. UserUnit is not assumed to be an inheritable page-tree attribute. Expected rejection counts as a pass only after the geometry validator is reached and raises its declared Ineligible reason; an unrelated parser failure cannot silently pass.', '',
        '| Case | Expected | Observed | Validation result |', '|---|---|---|---|']
    for row in result['cases']:
        lines.append(f"| {row['case']} | {'accept' if row['expected_geometry_accepted'] else 'reject'} | {'accept' if row['geometry_accepted'] else 'reject'} | {row.get('error', 'accepted')} |")
    lines += ['', 'This checks the PDF-library boundary, not the correctness of any source annotation or transform. No corpus documents or model outputs are consumed. The earlier census affected by NumberObject-versus-int rejection must be replaced with a fresh census; these checks do not repair its counts.', '',
              'The command exits nonzero if any expected outcome or rejection reason disagrees.']
    return '\n'.join(lines) + '\n'


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = run()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    args.output.with_suffix('.md').write_text(render_review(result, args.output), encoding='utf-8')
    for row in result["cases"]:
        print(row["case"], "accepted=" + str(row["geometry_accepted"]),
              "expected=" + str(row["expected_geometry_accepted"]), row.get("error", ""))
    print("DISAGREEMENTS", result["disagreements"])
    return 1 if result['disagreements'] else 0


if __name__ == "__main__":
    raise SystemExit(main())
