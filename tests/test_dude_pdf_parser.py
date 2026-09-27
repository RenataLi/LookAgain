"""Actual PDF serialization/reopening regressions for the parser boundary.

Run with a Python runtime that includes pypdf. These tests use unittest so the
bundled document runtime can run them without installing pytest or a package.
"""
from __future__ import annotations

import importlib.util
import io
from pathlib import Path
import unittest

try:
    from pypdf import PdfWriter
    from pypdf.generic import (
        ArrayObject, BooleanObject, FloatObject, NameObject, NumberObject,
        TextStringObject,
    )
except ImportError:
    PdfWriter = None

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "dude_pdf_parser_regression", ROOT / "experiments/prepare_dude_replication.py"
)
adapter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(adapter)
ABSENT = object()


@unittest.skipIf(PdfWriter is None, "pypdf required; run in bundled document runtime")
class PdfParserBoundaryTests(unittest.TestCase):
    def pdf(self, *, rotation=ABSENT, inherited=False, user_unit=ABSENT,
            box=None, crop=None):
        writer = PdfWriter()
        page = writer.add_blank_page(width=720, height=432)
        if rotation is not ABSENT:
            owner = writer._pages.get_object() if inherited else page
            owner[NameObject("/Rotate")] = rotation
        if user_unit is not ABSENT:
            page[NameObject("/UserUnit")] = user_unit
        if box is not None:
            page[NameObject("/MediaBox")] = ArrayObject(box)
        if crop is not None:
            page[NameObject("/CropBox")] = ArrayObject(crop)
        stream = io.BytesIO()
        writer.write(stream)
        pages, _ = adapter._pdf_metadata(stream.getvalue())
        self.assertEqual(len(pages), 1)
        return pages[0]

    def validate(self, page):
        return adapter.validate_page_geometry(
            page, {"page": 1, "width": 10, "height": 6, "unit": "inch"},
            [0, 0, 2000, 1200], 0,
        )

    def test_absent_and_explicit_numberobject_zero_are_identical(self):
        absent = self.pdf()
        explicit = self.pdf(rotation=NumberObject(0))
        self.assertEqual(absent, explicit)
        self.assertIs(type(explicit["rotation"]), int)
        self.assertEqual(self.validate(explicit)["render_size"], [2000, 1200])

    def test_inherited_zero_is_resolved_and_normalized(self):
        page = self.pdf(rotation=NumberObject(0), inherited=True)
        self.assertEqual(page, self.pdf())
        self.validate(page)

    def test_integral_float_zero_is_not_a_fractional_rotation(self):
        page = self.pdf(rotation=FloatObject(0.0))
        self.assertIs(type(page["rotation"]), int)
        self.assertEqual(page["rotation"], 0)
        self.validate(page)

    def test_nonzero_rotation_rejected(self):
        for value in (90, 180, 270, -90, 360):
            with self.subTest(value=value):
                page = self.pdf(rotation=NumberObject(value))
                self.assertIs(type(page["rotation"]), int)
                with self.assertRaisesRegex(adapter.Ineligible, "unsupported_pdf_rotation"):
                    self.validate(page)

    def test_inherited_nonzero_rotation_rejected(self):
        with self.assertRaisesRegex(adapter.Ineligible, "unsupported_pdf_rotation"):
            self.validate(self.pdf(rotation=NumberObject(90), inherited=True))

    def test_fractional_rotation_not_truncated(self):
        for value in (0.5, -0.5, 90.25):
            with self.subTest(value=value):
                page = self.pdf(rotation=FloatObject(value))
                self.assertIsNone(page["rotation"])
                with self.assertRaisesRegex(adapter.Ineligible, "unsupported_pdf_rotation"):
                    self.validate(page)

    def test_pdf_boolean_rotation_rejected(self):
        for value in (False, True):
            with self.subTest(value=value):
                page = self.pdf(rotation=BooleanObject(value))
                self.assertIsNone(page["rotation"])
                with self.assertRaisesRegex(adapter.Ineligible, "unsupported_pdf_rotation"):
                    self.validate(page)

    def test_numeric_string_rotation_not_coerced(self):
        page = self.pdf(rotation=TextStringObject("0"))
        self.assertIsNone(page["rotation"])
        with self.assertRaisesRegex(adapter.Ineligible, "unsupported_pdf_rotation"):
            self.validate(page)

    def test_valid_user_unit_library_scalars(self):
        for value in (NumberObject(1), FloatObject(1.0)):
            with self.subTest(value=value):
                page = self.pdf(user_unit=value)
                self.assertIs(type(page["user_unit"]), float)
                self.validate(page)

    def test_invalid_user_unit_not_coerced(self):
        for value in (BooleanObject(True), TextStringObject("1"), FloatObject(1.5)):
            with self.subTest(value=value):
                with self.assertRaisesRegex(adapter.Ineligible, "unsupported_pdf_user_unit"):
                    self.validate(self.pdf(user_unit=value))

    def test_valid_mixed_pdf_number_boxes(self):
        box = [NumberObject(0), FloatObject(0), NumberObject(720), FloatObject(432)]
        page = self.pdf(box=box, crop=box)
        self.assertEqual(page["mediabox"], [0.0, 0.0, 720.0, 432.0])
        self.assertTrue(all(type(v) is float for v in page["mediabox"]))
        self.validate(page)

    def test_box_boolean_or_string_not_coerced(self):
        for value in (BooleanObject(False), TextStringObject("0")):
            for field in ("box", "crop"):
                with self.subTest(value=value, field=field):
                    box = [value, NumberObject(0), NumberObject(720), NumberObject(432)]
                    page = self.pdf(**{field: box})
                    with self.assertRaisesRegex(adapter.Ineligible, "unsupported_pdf_boxes"):
                        self.validate(page)

    def test_numeric_subclass_and_invalid_python_scalar_boundary(self):
        class IntegerSubclass(int):
            pass
        self.assertIs(type(adapter._pdf_number(IntegerSubclass(0), integer=True)), int)
        for value in (True, False, "0", float("nan"), float("inf"), -float("inf")):
            with self.subTest(value=value):
                self.assertIsNone(adapter._pdf_number(value, integer=True))
                self.assertIsNone(adapter._pdf_number(value))


if __name__ == "__main__":
    unittest.main()
